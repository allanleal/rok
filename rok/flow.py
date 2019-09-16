import firedrake as fire
from firedrake import dot, div, grad, inner, curl, dx, ds
from .utils import boundaryNameToIndex, vectorComponentNameToIndex


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


    def addVelocityBC(self, value, onboundary):
        self.bcs_u.append((value, boundaryNameToIndex(onboundary), None))


    def addVelocityComponentBC(self, value, component, onboundary):
        self.bcs_u.append((value, boundaryNameToIndex(onboundary), vectorComponentNameToIndex(component)))


class DarcySolver:

    def __init__(self, problem):
        self.problem = problem

        mesh = problem.mesh
        rho = problem.rho
        mu = problem.mu
        k = problem.k
        f = problem.f
        bcs_p = problem.bcs_p
        bcs_u = problem.bcs_u

        U = fire.VectorFunctionSpace(mesh, 'CG', 1)
        V = fire.FunctionSpace(mesh, 'CG', 1)
        W = U * V

        self.solution = fire.Function(W)
        self.u = fire.Function(U, name='u')
        self.p = fire.Function(V, name='p')

        q, p = fire.TrialFunctions(W)
        w, v = fire.TestFunctions(W)

        n = fire.FacetNormal(mesh)

        # Stabilizing parameters
        h = fire.CellDiameter(mesh)
        has_mesh_characteristic_length = False
        delta_0 = fire.Constant(-1)
        delta_1 = fire.Constant(1/2)
        delta_2 = fire.Constant(1/2)
        delta_3 = fire.Constant(1/2)

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
        inv_kappa = 1.0/kappa

        # Classical mixed terms
        self.a = (dot(inv_kappa * q, w) - div(w) * p - delta_0 * v * div(q)) * dx
        self.L = -delta_0 * f * v * dx

        # Add the contributions of the pressure boundary conditions to L
        for pboundary, iboundary in bcs_p:
            self.L -= pboundary * dot(w, n) * ds(iboundary)

        # Stabilizing terms
        self.a += delta_1 * inner(kappa * (inv_kappa * q + grad(p)), delta_0 * inv_kappa * w + grad(v)) * dx
        self.a += delta_2 * inv_kappa * div(q) * div(w) * dx
        self.a += delta_3 * inner(kappa * curl(inv_kappa * q), curl(inv_kappa * w)) * dx
        self.L += delta_2 * inv_kappa * f * div(w) * dx

        # Construct the Dirichlet boundary conditions for velocity
        self.bcs = []
        for uboundary, iboundary, component in bcs_u:
            if component != None:
                self.bcs.append(fire.DirichletBC(W.sub(0).sub(component), rho*uboundary, iboundary))
            else:
                self.bcs.append(fire.DirichletBC(W.sub(0), rho*uboundary, iboundary))

        # solver_parameters = {
        #     # This setup is suitable for 3D
        #     'ksp_type': 'fgmres',
        #     'pc_type': 'ilu',
        #     'mat_type': 'aij',
        #     'ksp_rtol': 1e-5,
        #     'ksp_max_it': 2000,
        #     'ksp_monitor': None
        # }
        self.solver_parameters = {
            "mat_type": "aij",
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps"
        }

        lvp = fire.LinearVariationalProblem(self.a, self.L, self.solution, bcs=self.bcs)

        self.solver = fire.LinearVariationalSolver(lvp, solver_parameters=self.solver_parameters)


    def solve(self):
        self.solver.solve()
        self.p.assign(self.solution.sub(1))
        self.u.assign(self.solution.sub(0))
        self.u /= self.problem.rho
