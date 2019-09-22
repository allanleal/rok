from firedrake import *


class _TransportSolver(object):

    def __init__(self, method='dg'):
        # Initialize default values for the velocity, diffusion and source parameters
        self.velocity = Constant(0.0)
        self.diffusion = Constant(0.0)
        self.source = Constant(0.0)
        self.method = method

        # Initialize the list of DirichletBC instances
        self.bcs = []

        # The time step as a dolfin.Constant instance to avoid compilation when its value changes
        self.dt = Constant(0.0)

        # The flag that indicates if the solver has been initialized before evolving the solution
        self.initialized = False

    def setVelocity(self, velocity):
        self.velocity = velocity
        self.initialized = False

    def setDiffusion(self, diffusion):
        self.diffusion = diffusion
        self.initialized = False

    def setSource(self, source):
        self.source = source
        self.initialized = False

    def setBoundaryConditions(self, bcs):
        self.bcs = bcs if hasattr(bcs, '__len__') else [bcs]

    def initialize(self, function_space):
        self.initialized = True

        V = function_space
        self.u = Function(V)

        if self.method == 'supg':
            self.a, self.L = self._supg_form(V)
        elif self.method == 'dg':
            self.a, self.L = self._primal_dg_advection(V)
            # self.a, self.L = self._improved_dg_advection(V)  # Work in progress (don't use it)
        else:
            raise ValueError(f'Invalid method for transport problem. Method provided {self.method} is not available.')

        self.problem = LinearVariationalProblem(self.a, self.L, self.u, bcs=self.bcs)
        self.solver = LinearVariationalSolver(self.problem)

    def _supg_form(self, function_space):
        velocity = self.velocity
        diffusion = self.diffusion
        source = self.source
        dt = self.dt

        mesh = function_space.mesh()

        V = function_space

        u0 = self.u0 = Function(V)

        u = TrialFunction(V)
        v = TestFunction(V)

        # Semi-discrete residual
        r = (div(velocity * u) - div(diffusion * grad(u)) - source)

        # Galerkin variational problem
        F = v * (u - u0) * dx + dt * (v * div(velocity * u) * dx + \
                                      dot(grad(v), diffusion * grad(u)) * dx - source * v * dx)

        # Add SUPG stabilisation terms
        h_k = sqrt(2) * CellVolume(mesh) / CellDiameter(mesh)
        vnorm = sqrt(dot(velocity, velocity))
        m_k = 1.0 / 3.0
        Pe_k = m_k * vnorm * h_k / (2.0 * diffusion)
        one = Constant(1.0)
        eps_k = conditional(gt(Pe_k, one), one, Pe_k)
        tau_k = h_k / (2.0 * vnorm) * eps_k
        F += dt * tau_k * dot(velocity, grad(v)) * r * dx

        return lhs(F), rhs(F)

    def _primal_dg_advection(self, function_space):
        velocity = self.velocity
        diffusion = self.diffusion
        source = self.source
        dt = self.dt

        mesh = function_space.mesh()

        V = function_space

        u0 = self.u0 = Function(V)

        u = TrialFunction(V)
        v = TestFunction(V)

        n = FacetNormal(mesh)
        un = 0.5 * (dot(velocity, n) + abs(dot(velocity, n)))

        alpha = Constant(1e0)
        theta = Constant(0.5)
        h = CellDiameter(mesh)

        def a(u, v):
            a_int = dot(grad(v), diffusion * grad(u) - velocity * u) * dx
            a_velocity = dot(jump(v), un('+') * u('+') - un('-') * u('-')) * dS + dot(v, un * u) * ds
            a_facets = diffusion * (alpha / avg(h)) * dot(jump(u, n), jump(v, n)) * dS \
                       - diffusion * dot(avg(grad(u)), jump(v, n)) * dS \
                       - diffusion * dot(jump(u, n), avg(grad(v))) * dS
            a = a_int + a_facets + a_velocity
            return a

        # Define variational forms
        a0 = a(u0, v)
        a1 = a(u, v)

        A = (1 / dt) * inner(u, v) * dx - (1 / dt) * inner(u0, v) * dx + theta * a1 + (1 - theta) * a0
        L_rhs = dt * source * v * dx
        F = A - L_rhs

        return lhs(F), rhs(F)

    # Don't use this one. Work in progress.
    def _improved_dg_advection(self, function_space):
        velocity = self.velocity
        diffusion = self.diffusion
        source = self.source
        dt = self.dt

        mesh = function_space.mesh()

        V = function_space

        u0 = self.u0 = Function(V)

        u = TrialFunction(V)
        v = TestFunction(V)

        n = FacetNormal(mesh)
        un = 0.5 * (dot(velocity, n) + abs(dot(velocity, n)))

        alpha = Constant(1e0)
        theta = Constant(1.0)
        h = CellDiameter(mesh)

        def a(u, v):
            a_int = dot(grad(v), diffusion * grad(u)) * dx
            # Internal facets
            a_facets = diffusion * (alpha / avg(h)) * dot(jump(u, n), jump(v, n)) * dS \
                       - diffusion * dot(avg(grad(u)), jump(v, n)) * dS \
                       - diffusion * dot(jump(u, n), avg(grad(v))) * dS #\
                       # - conditional(
                       #      dot(velocity, n) < 0, dot(diffusion('+') * grad(u)('+'), jump(v, n)), 0
                       # ) * ds \
                       # - conditional(
                       #      dot(velocity, n) > 0, dot(diffusion('-') * grad(u)('-'), jump(v, n)), 0
                       # ) * ds \
                       # - conditional(
                       #      dot(velocity, n) < 0, dot(diffusion('+') * grad(v)('+'), jump(u, n)), 0
                       # ) * ds \
                       # - conditional(
                       #      dot(velocity, n) > 0, dot(diffusion('-') * grad(v)('-'), jump(u, n)), 0
                       # ) * ds
            # External facets
            a_facets += (alpha / h) * dot(jump(u, n), jump(v, n)) * ds \
                        - dot(avg(diffusion * grad(u)), jump(v, n)) * ds \
                        - dot(avg(diffusion * grad(v)), jump(u, n)) * ds

            a_velocity = dot(jump(v), un('+') * u('+') - un('-') * u('-')) * dS + dot(v, un * u) * ds

            a = a_int + a_facets + a_velocity
            return a

        # Define variational forms
        a0 = a(u0, v)
        a1 = a(u, v)

        A = (1 / dt) * inner(u, v) * dx - (1 / dt) * inner(u0, v) * dx + theta * a1 + (1 - theta) * a0
        # A = a(u, v)
        L_rhs = dt * source * v * dx
        F = A - L_rhs

        return lhs(F), rhs(F)

    def step(self, u, dt):
        if not self.initialized:
            self.initialize(u.function_space())
        self.dt.assign(dt)
        self.u0.assign(u)
        self.u.assign(u)
        self.solver.solve()
        u.assign(self.u)


class TransportSolver(object):
    def __init__(self, method='dg'):
        self.pimpl = _TransportSolver(method=method)

    def setVelocity(self, velocity):
        self.pimpl.setVelocity(velocity)

    def setDiffusion(self, diffusion):
        self.pimpl.setDiffusion(diffusion)

    def setSource(self, source):
        self.pimpl.setSource(source)

    def setBoundaryConditions(self, bc):
        self.pimpl.setBoundaryConditions(bc)

    def step(self, u, dt):
        self.pimpl.step(u, dt)
