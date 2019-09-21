import numpy as np
import rok


# Time constants
second = 1
minute = 60
hour = 60 * minute
day = 24 * hour
week = 7 * day
year = 365 * day

# Mathematical constants
pi = 3.14159265359

# Parameters for the reactive transport simulation
nx = 100          # the number of mesh cells along the x-axis
ny = 100          # the number of mesh cells along the y-axis
nz = 25           # the number of mesh cells along the y-axis
Lx = 1.6          # the length of the mesh along the x-axis
Ly = 1.0          # the length of the mesh along the y-axis
nsteps = 10000    # the number of time steps
cfl = 0.01        # the CFL number to be used in the calculation of time step

# method = 'cgls'
# method = 'dgls'
method = 'sdhm'

# The path to where the result files are output
resultsdir = 'results/demo-transport-in-heterogeneous-medium/{}/'.format(method)

# Initialise the mesh
mesh = rok.RectangleMesh(nx, ny, Lx, Ly, quadrilateral=True)

V = rok.FunctionSpace(mesh, "CG", 1)

x, y = rok.SpatialCoordinate(mesh)

# Model parameters
rho = rok.Constant(1000.0)  # water density (in units of kg/m3)
mu = rok.Constant(8.9e-4)  # water viscosity (in units of Pa*s)
k = rok.permeability(V, minval=1e-18, len_scale=20)  # Sandstone (?)
f = rok.Constant(0.0)  # the source rate in the flow calculation

D  = rok.Constant(1.0e-9)               # the diffusion coefficient (in units of m2/s)

c = rok.Function(V, name="Concentration")

cL = rok.Constant(1.0)

# Initialize the Darcy flow solver
problem = rok.DarcyProblem(mesh)
problem.setFluidDensity(rho)
problem.setFluidViscosity(mu)
problem.setRockPermeability(k)
problem.setSourceRate(f)
problem.addPressureBC(5e6, 'left')  # injection well
problem.addPressureBC(5e5, 'right')  # production well
problem.addVelocityComponentBC(rok.Constant(0.0), 'y', 'bottom')
problem.addVelocityComponentBC(rok.Constant(0.0), 'y', 'top')

flow = rok.DarcySolver(problem, method=method)
flow.solve()

rok.File(resultsdir + 'flow.pvd').write(flow.u, flow.p, k)

# Initialize the transport solver
transport = rok.TransportSolver()
transport.setVelocity(flow.u)
transport.setDiffusion(D)
transport.setSource(f)

# bc = rok.DirichletBC(V, rok.Constant(1.0), 1)
bc = rok.DirichletBC(V, cL, 1)
transport.setBoundaryConditions([bc])

t = 0.0
step = 0

file = rok.File(resultsdir + 'concentration.pvd')
file.write(c)

max_ux = np.max(flow.u.dat.data[:,0])
max_uy = np.max(flow.u.dat.data[:,1])
delta_x = Lx/nx
delta_y = Ly/nx

dt = cfl/max(max_ux/delta_x, max_uy/delta_y)

print('max(u) = ', np.max(flow.u.dat.data[:,0]))
print('max(k) = ', np.max(k.dat.data))

print('div(u)*dx =', rok.assemble(rok.div(flow.u)*rok.dx))
print('dt = {} minute'.format(dt/minute))

while step <= nsteps:
    print('Time: {:<5.2f} day ({}/{})'.format(t/day, step, nsteps))

    transport.step(c, dt)

    t += dt
    step += 1

    if step % 50 == 0:
        file.write(c)
