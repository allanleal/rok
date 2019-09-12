from firedrake import *


class _TransportSolver(object):

    def __init__(self):

        # Initialize default values for the velocity, diffusion and source parameters
        self.velocity = Constant(0.0)
        self.diffusion = Constant(0.0)
        self.source = Constant(0.0)

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

        velocity = self.velocity
        diffusion = self.diffusion
        source = self.source
        dt = self.dt

        V = function_space

        mesh = V.mesh()

        u0 = self.u0 = Function(V)

        u = TrialFunction(V)
        v = TestFunction(V)

        # Residual
        r = u - u0 + dt*(div(velocity * u) - \
            div(diffusion*grad(u)) - source)

        # Galerkin variational problem
        F = v*(u - u0)*dx + dt*(v*div(velocity * u)*dx + \
            dot(grad(v), diffusion*grad(u))*dx - source*v*dx)

        # Add SUPG stabilisation terms
        h_k = sqrt(2) * CellVolume(mesh) / CellDiameter(mesh)
        vnorm = sqrt(dot(velocity, velocity))
        m_k = 1.0 / 3.0
        Pe_k = m_k * vnorm * h_k / (2.0 * diffusion)
        one = Constant(1.0)
        eps_k = conditional(gt(Pe_k, one), one, Pe_k)
        tau_k = h_k / (2.0 * vnorm) * eps_k
        F += tau_k*dot(velocity, grad(v))*r*dx

        # Create bilinear and linear forms
        self.a = lhs(F)
        self.L = rhs(F)

        self.A = assemble(self.a, bcs=self.bcs)
        self.b = assemble(self.L, bcs=self.bcs)

        self.u = Function(V)

        self.problem = LinearVariationalProblem(self.a, self.L, self.u, bcs=self.bcs)
        self.solver = LinearVariationalSolver(self.problem)


    def step(self, u, dt):
        if not self.initialized:
            self.initialize(u.function_space())
        self.dt.assign(dt)
        self.u0.assign(u)
        self.u.assign(u)
        self.solver.solve()
        u.assign(self.u)


class TransportSolver(object):
    def __init__(self):
        self.pimpl = _TransportSolver()


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

