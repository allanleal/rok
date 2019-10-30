import firedrake as fire
import rok

method = "supg"

nx = 25  # the number of mesh cells along the x-coordinate
ny = 25  # the number of mesh cells along the y-coordinate

nsteps = 100  # the number of time steps

D = fire.Constant(1.0e-9)  # the diffusion coefficient (in units of m2/s)
v = fire.Constant([1.0e-6, 0.0])  # the fluid pore velocity (in units of m/s)
q = fire.Constant(0.0)  # the source rate (in units of [u]/s)
dt = 2.0e5  # the time step (in units of s)

# Initialise the mesh
mesh = fire.UnitSquareMesh(nx, ny, quadrilateral=True)

if method == "supg":
    finite_element_space = "CG"
elif method == "dg":
    finite_element_space = "DG"
else:
    raise ValueError(f"Method {method} is unavailable.")

V = fire.FunctionSpace(mesh, finite_element_space, 1)

bc = fire.DirichletBC(V, fire.Constant(1.0), 1)

u = fire.Function(V)

# Initialize the transport solver
transport = rok.TransportSolver(method=method)
transport.setVelocity(v)
transport.setDiffusion(D)
transport.setSource(q)
transport.setBoundaryConditions([bc])

t = 0.0
step = 0

outfile = fire.File("results_demo-transport/u.pvd")
outfile.write(u)

while step < nsteps:

    transport.step(u, dt)

    t += dt
    step += 1

    if step % 1 == 0:
        outfile.write(u)
        print("t = {:<15.3f} step = {:<15}".format(t, step))
