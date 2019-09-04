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


    def initialize(self, u):
        self.initialized = True

        velocity = self.velocity
        diffusion = self.diffusion
        source = self.source
        dt = self.dt

        V = u.function_space()
        mesh = V.mesh()

        u0 = u

        u = TrialFunction(V)
        v = TestFunction(V)

        # Mid-point solution
        u_mid = 0.5*(u0 + u)

        # Residual
        r = u - u0 + dt*(div(velocity * u_mid) - \
            div(diffusion*grad(u_mid)) - source)

        # Galerkin variational problem
        F = v*(u - u0)*dx + dt*(v*div(velocity * u_mid)*dx + \
            dot(grad(v), diffusion*grad(u_mid))*dx - source*v*dx)

        # Add SUPG stabilisation terms
        h = CellDiameter(mesh)
        vnorm = sqrt(dot(velocity, velocity))
        tau = h/(2.0*vnorm)

        F += tau*dot(velocity, grad(v))*r*dx

        # Create bilinear and linear forms
        self.a = lhs(F)
        self.L = rhs(F)

        self.A = assemble(self.a, bcs=self.bcs)
        self.b = assemble(self.L, bcs=self.bcs)


    def step(self, u, dt):
        if not self.initialized:
            self.initialize(u)
        self.dt.assign(dt)
        assemble(self.a, tensor=self.A, bcs=self.bcs)
        assemble(self.L, tensor=self.b, bcs=self.bcs)
        solve(self.A, u, self.b)


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


# Discretization parameters
ncells_x  = 40
ncells_y  = 40

# Problem parameters
velocity  = 1.0e-2
diffusion = 1.0e-3
diffusion = 0.0
source    = 0.0

# Time integration parameters
cfl       = 0.1
T         = 100
# dt        = cfl * (1.0/ncells_x) / velocity  # v*dt/dx = CFL
dt        = 1.0


velocity  = Constant((velocity, 0.0))
diffusion = Constant(diffusion)
source    = Constant(source)


# mesh = UnitSquareMesh(ncells_x, ncells_y)
mesh = UnitSquareMesh(ncells_x, ncells_y, quadrilateral=True)

V = FunctionSpace(mesh, "CG", 1)

bc = DirichletBC(V, Constant(1.0), 1)

u = Function(V)

transport = TransportSolver()
transport.setVelocity(velocity)
transport.setDiffusion(diffusion)
transport.setSource(source)
transport.setBoundaryConditions([bc])

t = 0.0

step = 0

outfile = File("u.pvd")
outfile.write(u)

while t < T:

  transport.step(u, dt)

  umin = min(u.dat.data)
  umax = max(u.dat.data)

  t += dt
  step += 1

  if step % 5 == 0:
    outfile.write(u)
    print("t = {:<15.3f} step = {:<15} umin = {:<15.3f} umax = {:<15.3f}".format(t, step, umin, umax))


