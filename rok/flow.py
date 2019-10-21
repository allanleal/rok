import firedrake as fire
from firedrake import dot, div, grad, inner, curl, dx, ds, dS, jump, avg
from .utils import boundaryNameToIndex, vectorComponentNameToIndex, DirichletExpressionBC


class DarcyProblem:

    def __init__(self, mesh):
        self.mesh = mesh
        self.rho = None
        self.mu = None
        self.k = None
        self.f = None
        self.bcs_p = []
        self.bcs_u = []

    def setFluidViscosity(self, mu):
        self.mu = mu

    def setFluidDensity(self, rho):
        self.rho = rho

    def setRockPermeability(self, k):
        self.k = k

    def setSourceRate(self, f):
        self.f = f

    def addImpermeableBoundary(self, index):
        pass

    def addPressureBC(self, value, onboundary):
        self.bcs_p.append((value, boundaryNameToIndex(onboundary)))

    def addVelocityBC(self, vecvalue, onboundary):
        if type(vecvalue) in [list, tuple]:
            vecvalue = fire.Constant(vecvalue)
        self.bcs_u.append((vecvalue, boundaryNameToIndex(onboundary), None))

    def addVelocityComponentBC(self, value, component, onboundary):
        if type(value) in [float, int]:
            value = fire.Constant(value)
        self.bcs_u.append((value, boundaryNameToIndex(onboundary), vectorComponentNameToIndex(component)))


class DarcySolver:

    def __init__(self, problem, u_degree=1, p_degree=1, lambda_degree=1, method='cgls'):
        self.problem = problem
        self.method = method
        mesh = problem.mesh
        bcs_p = problem.bcs_p
        bcs_u = problem.bcs_u

        if method == 'cgls':
            pressure_family = 'CG'
            velocity_family = 'CG'
            dirichlet_method = 'topological'
        elif method == 'dgls':
            pressure_family = 'DG'
            velocity_family = 'DG'
            dirichlet_method = 'geometric'
        elif method == 'sdhm':
            pressure_family = 'DG'
            velocity_family = 'DG'
            trace_family = 'HDiv Trace'
            dirichlet_method = 'topological'
        else:
            raise ValueError(f'Invalid FEM for solving Darcy Flow. Method provided: {method}')

        self._U = fire.VectorFunctionSpace(mesh, velocity_family, u_degree)
        self._V = fire.FunctionSpace(mesh, pressure_family, p_degree)

        if method == 'cgls' or method == 'dgls':
            self._W = self._U * self._V
        if method == 'sdhm':
            self._T = fire.FunctionSpace(mesh, trace_family, lambda_degree)
            self._W = self._U * self._V * self._T

        self.solution = fire.Function(self._W)
        if method == 'cgls':
            self.u, self.p = self.solution.split()
            self._a, self._L = self.cgls_form(problem, mesh, bcs_p)
        if method == 'dgls':
            self.u, self.p = self.solution.split()
            self._a, self._L = self.dgls_form(problem, mesh, bcs_p)
        if method == 'sdhm':
            self.u, self.p, self.tracer = self.solution.split()
            self._a, self._L = self.sdhm_form(problem, mesh, bcs_p, bcs_u)

        self.u.rename('u', 'label')
        self.p.rename('p', 'label')
        if method == 'sdhm':
            self.solver_parameters = {
                'snes_type': 'ksponly',
                'mat_type': 'matfree',
                'pmat_type': 'matfree',
                'ksp_type': 'preonly',
                'pc_type': 'python',
                # Use the static condensation PC for hybridized problems
                # and use a direct solve on the reduced system for lambda_h
                'pc_python_type': 'firedrake.SCPC',
                'pc_sc_eliminate_fields': '0, 1',
                'condensed_field': {
                    'ksp_type': 'preonly',
                    'pc_type': 'lu',
                    'pc_factor_mat_solver_type': 'mumps'
                }
            }

            F = self._a - self._L
            nvp = fire.NonlinearVariationalProblem(F, self.solution)

            self.solver = fire.NonlinearVariationalSolver(nvp, solver_parameters=self.solver_parameters)
        else:
            self.bcs = []
            rho = self.problem.rho
            for uboundary, iboundary, component in bcs_u:
                if component is not None:
                    self.bcs.append(
                        DirichletExpressionBC(self._W.sub(0).sub(component), rho*uboundary, iboundary, method=dirichlet_method)
                    )
                else:
                    self.bcs.append(
                        DirichletExpressionBC(self._W.sub(0), rho*uboundary, iboundary, method=dirichlet_method)
                    )

            # self.solver_parameters = {
            #     # This setup is suitable for 3D
            #     'ksp_type': 'lgmres',
            #     'pc_type': 'lu',
            #     'mat_type': 'aij',
            #     'ksp_rtol': 1e-8,
            #     'ksp_max_it': 2000,
            #     'ksp_monitor': None
            # }
            self.solver_parameters = {
                "mat_type": "aij",
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps"
            }

            lvp = fire.LinearVariationalProblem(self._a, self._L, self.solution, bcs=self.bcs)

            self.solver = fire.LinearVariationalSolver(lvp, solver_parameters=self.solver_parameters)

    def cgls_form(self, problem, mesh, bcs_p):
        rho = problem.rho
        mu = problem.mu
        k = problem.k
        f = problem.f

        q, p = fire.TrialFunctions(self._W)
        w, v = fire.TestFunctions(self._W)

        n = fire.FacetNormal(mesh)

        # Stabilizing parameters
        h = fire.CellDiameter(mesh)
        has_mesh_characteristic_length = True
        delta_0 = fire.Constant(1)
        delta_1 = fire.Constant(-1 / 2)
        delta_2 = fire.Constant(1 / 2)
        delta_3 = fire.Constant(1 / 2)

        # Some good stabilizing methods that I use:
        # 1) CLGS (Correa and Loula method, it's a Galerkin Least-Squares residual formulation):
        #   * delta_0 = 1
        #   * delta_1 = -1/2
        #   * delta_2 = 1/2
        #   * delta_3 = 1/2
        # 2) CLGS (Div):
        #   * delta_0 = 1
        #   * delta_1 = -1/2
        #   * delta_2 = 1/2
        #   * delta_3 = 0
        # 3) Original Hughes's adjoint (variational multiscale) method (HVM):
        #   * delta_0 = -1
        #   * delta_1 = 1/2
        #   * delta_2 = 0
        #   * delta_3 = 0
        # 4) HVM (Div):
        #   * delta_0 = -1
        #   * delta_1 = 1/2
        #   * delta_2 = 1/2
        #   * delta_3 = 0
        # 5) Enhanced HVM (eHVM, this one is proposed by me. It was never published before):
        #   * delta_0 = -1
        #   * delta_1 = 1/2
        #   * delta_2 = 1/2
        #   * delta_3 = 1/2

        # I'm currently investigating these modifications in my thesis. They work good for DG methods.
        if has_mesh_characteristic_length:
            delta_2 = delta_2 * h * h
            delta_3 = delta_3 * h * h

        kappa = rho * k / mu
        inv_kappa = 1.0 / kappa

        # Classical mixed terms
        a = (dot(inv_kappa * q, w) - div(w) * p - delta_0 * v * div(q)) * dx
        L = -delta_0 * f * v * dx

        # Add the contributions of the pressure boundary conditions to L
        for pboundary, iboundary in bcs_p:
            L -= pboundary * dot(w, n) * ds(iboundary)

        # Stabilizing terms
        a += delta_1 * inner(kappa * (inv_kappa * q + grad(p)), delta_0 * inv_kappa * w + grad(v)) * dx
        a += delta_2 * inv_kappa * div(q) * div(w) * dx
        a += delta_3 * inner(kappa * curl(inv_kappa * q), curl(inv_kappa * w)) * dx
        L += delta_2 * inv_kappa * f * div(w) * dx

        return a, L

    def dgls_form(self, problem, mesh, bcs_p):
        rho = problem.rho
        mu = problem.mu
        k = problem.k
        f = problem.f

        q, p = fire.TrialFunctions(self._W)
        w, v = fire.TestFunctions(self._W)

        n = fire.FacetNormal(mesh)
        h = fire.CellDiameter(mesh)

        # Stabilizing parameters
        has_mesh_characteristic_length = True
        delta_0 = fire.Constant(1)
        delta_1 = fire.Constant(-1 / 2)
        delta_2 = fire.Constant(1 / 2)
        delta_3 = fire.Constant(1 / 2)
        eta_p = fire.Constant(100)
        eta_q = fire.Constant(100)
        h_avg = (h('+') + h('-')) / 2.
        if has_mesh_characteristic_length:
            delta_2 = delta_2 * h * h
            delta_3 = delta_3 * h * h

        kappa = rho * k / mu
        inv_kappa = 1.0 / kappa

        # Classical mixed terms
        a = (dot(inv_kappa * q, w) - div(w) * p - delta_0 * v * div(q)) * dx
        L = -delta_0 * f * v * dx

        # DG terms
        a += jump(w, n) * avg(p) * dS - \
             avg(v) * jump(q, n) * dS

        # Edge stabilizing terms
        a += (eta_q * h_avg) * avg(inv_kappa) * (jump(q, n) * jump(w, n)) * dS + \
             (eta_p / h_avg) * avg(kappa) * dot(jump(v, n), jump(p, n)) * dS

        # Add the contributions of the pressure boundary conditions to L
        for pboundary, iboundary in bcs_p:
            L -= pboundary * dot(w, n) * ds(iboundary)

        # Stabilizing terms
        a += delta_1 * inner(kappa * (inv_kappa * q + grad(p)), delta_0 * inv_kappa * w + grad(v)) * dx
        a += delta_2 * inv_kappa * div(q) * div(w) * dx
        a += delta_3 * inner(kappa * curl(inv_kappa * q), curl(inv_kappa * w)) * dx
        L += delta_2 * inv_kappa * f * div(w) * dx

        return a, L

    def sdhm_form(self, problem, mesh, bcs_p, bcs_u):
        rho = problem.rho
        mu = problem.mu
        k = problem.k
        f = problem.f

        q, p, lambda_h = fire.split(self.solution)
        w, v, mu_h = fire.TestFunctions(self._W)

        n = fire.FacetNormal(mesh)
        h = fire.CellDiameter(mesh)

        # Stabilizing parameters
        has_mesh_characteristic_length = True
        beta_0 = fire.Constant(1e-15)
        delta_0 = fire.Constant(-1)
        delta_1 = fire.Constant(1 / 2)
        delta_2 = fire.Constant(0)
        delta_3 = fire.Constant(0)

        # h_avg = (h('+') + h('-')) / 2.
        beta = beta_0 / h
        beta_avg = beta_0 / h('+')
        if has_mesh_characteristic_length:
            delta_2 = delta_2 * h * h
            delta_3 = delta_3 * h * h

        kappa = rho * k / mu
        inv_kappa = 1.0 / kappa

        # Classical mixed terms
        a = (dot(inv_kappa * q, w) - div(w) * p - delta_0 * v * div(q)) * dx
        L = -delta_0 * f * v * dx

        # Hybridization terms
        a += lambda_h('+') * dot(w, n)('+') * dS + mu_h('+') * dot(q, n)('+') * dS
        a += beta_avg * kappa('+') * (lambda_h('+') - p('+')) * (mu_h('+') - v('+')) * dS

        # Add the contributions of the pressure boundary conditions to L
        primal_bc_markers = list(mesh.exterior_facets.unique_markers)
        for pboundary, iboundary in bcs_p:
            primal_bc_markers.remove(iboundary)
            a += (pboundary * dot(w, n) + mu_h * dot(q, n)) * ds(iboundary)
            a += beta * kappa * (lambda_h - pboundary) * mu_h * ds(iboundary)

        unprescribed_primal_bc = primal_bc_markers
        for bc_marker in unprescribed_primal_bc:
            a += (lambda_h * dot(w, n) + mu_h * dot(q, n)) * ds(bc_marker)
            a += beta * kappa * lambda_h * mu_h * ds(bc_marker)

        # Add the (weak) contributions of the velocity boundary conditions to L
        for uboundary, iboundary, component in bcs_u:
            if component is not None:
                dim = mesh.geometric_dimension()
                bc_array = []
                for _ in range(dim):
                    bc_array.append(0.0)
                bc_array[component] = uboundary
                bc_as_vector = fire.Constant(bc_array)
                L += mu_h * dot(bc_as_vector, n) * ds(iboundary)
            else:
                L += mu_h * dot(uboundary, n) * ds(iboundary)

        # Stabilizing terms
        a += delta_1 * inner(kappa * (inv_kappa * q + grad(p)), delta_0 * inv_kappa * w + grad(v)) * dx
        a += delta_2 * inv_kappa * div(q) * div(w) * dx
        a += delta_3 * inner(kappa * curl(inv_kappa * q), curl(inv_kappa * w)) * dx
        L += delta_2 * inv_kappa * f * div(w) * dx

        return a, L

    def solve(self):
        if not self.method == 'sdhm':  # TODO: temporary fix. It has to be improved.
            for bc in self.bcs:
                bc.update()
        self.solver.solve()
        self.p.assign(self.solution.sub(1))
        self.u.assign(self.solution.sub(0))
        for i in range(self.u.dat.data.shape[1]):
            self.u.dat.data[:, i] /= self.problem.rho.dat.data
