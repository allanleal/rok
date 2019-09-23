import numpy as np
import rok
import matplotlib.pyplot as plt

# Time constants
second = 1
minute = 60
hour = 60 * minute
day = 24 * hour
week = 7 * day
year = 365 * day

# Parameters for the flow simulation
method = 'dgls'
nx = 100          # the number of mesh cells along the x-axis
ny = 100          # the number of mesh cells along the y-axis
nz = 25           # the number of mesh cells along the y-axis
Lx = 1e3          # the length of the mesh along the x-axis
Ly = 1e3          # the length of the mesh along the y-axis
nsteps = 5000    # the number of time steps
cfl = 0.3        # the CFL number to be used in the calculation of time step

# The path to where the result files are output
resultsdir = f'results/demo-five-spot-in-heterogeneous-medium/{method}/'

# Initialise the mesh
mesh = rok.RectangleMesh(nx, ny, Lx, Ly, quadrilateral=True)

V = rok.FunctionSpace(mesh, "DG", 0)

x, y = rok.SpatialCoordinate(mesh)

# Model parameters
rho = rok.Constant(1000.0)  # water density (in units of kg/m3)
mu = rok.Constant(8.9e-4)  # water viscosity (in units of Pa*s)
k = rok.permeability(V, minval=1e-20, len_scale=25, seed=666)
f = rok.Constant(0.0)  # the source rate in the flow calculation

# Boundary conditions
h_well = rok.Constant(0.01 * Lx)

# Velocity BC?
# v_left = rok.conditional(y < h_well, -rok.sqrt(rok.Constant(2)) / (3 * (h_well - y)), 0)
# v_bottom = rok.conditional(x < h_well, -rok.sqrt(rok.Constant(2)) / (3 * (h_well - x)), 0)
# v_right = rok.conditional(Ly - y < h_well, rok.sqrt(rok.Constant(2)) / (3 * (y - h_well)), 0)
# v_top = rok.conditional(Lx - x < h_well, rok.sqrt(rok.Constant(2)) / (3 * (x - h_well)), 0)

v_left = rok.conditional(y < h_well, 1e-4, 0)
v_bottom = rok.conditional(x < h_well, 1e-4, 0)
v_right = rok.conditional(Ly - y < h_well, 1e-4, 0)
v_top = rok.conditional(Lx - x < h_well, 1e-4, 0)

# Pressure BC?
# p_left = rok.conditional(y < h_well, 1e5, 0)
# p_bottom = rok.conditional(x < h_well, 1e5, 0)
# p_right = rok.conditional(Ly - y < h_well, -1e5, 0)
# p_top = rok.conditional(Lx - x < h_well, -1e5, 0)

# Initialize the Darcy flow solver
problem = rok.DarcyProblem(mesh)
problem.setFluidDensity(rho)
problem.setFluidViscosity(mu)
problem.setRockPermeability(k)
problem.setSourceRate(f)
# problem.addPressureBC(p_left, 'left')
# problem.addPressureBC(p_bottom, 'bottom')
# problem.addPressureBC(p_right, 'right')
# problem.addPressureBC(p_top, 'top')
problem.addVelocityComponentBC(v_left, 'x', 'left')
problem.addVelocityComponentBC(v_bottom, 'y', 'bottom')
problem.addVelocityComponentBC(v_right, 'x', 'right')
problem.addVelocityComponentBC(v_top, 'y', 'top')

flow = rok.DarcySolver(problem, method=method)
flow.solve()

rok.File(resultsdir + 'flow.pvd').write(flow.u, flow.p, k)

# Initialize the transport solver
V_transport = rok.FunctionSpace(mesh, "DG", 1)
c = rok.Function(V_transport, name="Concentration")
D = rok.Constant(0.0e-9)               # the diffusion coefficient (in units of m2/s)
transport = rok.TransportSolver(method='dg')
transport.setVelocity(flow.u)
transport.setDiffusion(D)
transport.setSource(f)

# bc = rok.DirichletBC(V, rok.Constant(1.0), 1)
# cL = rok.Constant(1.0)
c_left = rok.conditional(y < h_well, 1, 0)
c_bottom = rok.conditional(x < h_well, 1, 0)
bc_left = rok.DirichletBC(V_transport, c_left, 1, method='geometric')
bc_bottom = rok.DirichletBC(V_transport, c_bottom, 3, method='geometric')
transport.setBoundaryConditions([bc_left, bc_bottom])

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

#
# rok.plot(k)
# plt.axis('off')
# plt.show()
#
# rok.plot(flow.u)
# plt.axis('off')
# plt.show()
#
# rok.plot(flow.p)
# plt.axis('off')
# plt.show()
