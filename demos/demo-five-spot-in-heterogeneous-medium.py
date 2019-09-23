import numpy as np
import rok
import matplotlib.pyplot as plt

# Parameters for the flow simulation
method = 'dgls'
nx = 100          # the number of mesh cells along the x-axis
ny = 100          # the number of mesh cells along the y-axis
nz = 25           # the number of mesh cells along the y-axis
Lx = 1.0          # the length of the mesh along the x-axis
Ly = 1.0          # the length of the mesh along the y-axis

# The path to where the result files are output
resultsdir = f'results/demo-five-spot-in-heterogeneous-medium/{method}/'

# Initialise the mesh
mesh = rok.RectangleMesh(nx, ny, Lx, Ly, quadrilateral=True)

V = rok.FunctionSpace(mesh, "DG", 0)

x, y = rok.SpatialCoordinate(mesh)

# Model parameters
rho = rok.Constant(1000.0)  # water density (in units of kg/m3)
mu = rok.Constant(8.9e-4)  # water viscosity (in units of Pa*s)
k = rok.permeability(V, minval=1e-18, len_scale=20)
f = rok.Constant(0.0)  # the source rate in the flow calculation

# Boundary conditions
# h_well = rok.sqrt(rok.Constant(2)) / (3 * (Lx / nx))
h_well = rok.Constant(0.025 * Lx)
# v_left = rok.conditional(y < h_well, -rok.sqrt(rok.Constant(2)) / (3 * (h_well - y)), 0)
# v_bottom = rok.conditional(x < h_well, -rok.sqrt(rok.Constant(2)) / (3 * (h_well - x)), 0)
# v_right = rok.conditional(Ly - y < h_well, rok.sqrt(rok.Constant(2)) / (3 * (y - h_well)), 0)
# v_top = rok.conditional(Lx - x < h_well, rok.sqrt(rok.Constant(2)) / (3 * (x - h_well)), 0)

v_left = rok.conditional(y < h_well, 1e-2, 0)
v_bottom = rok.conditional(x < h_well, 1e-2, 0)
v_right = rok.conditional(Ly - y < h_well, 1e-2, 0)
v_top = rok.conditional(Lx - x < h_well, 1e-2, 0)

# Initialize the Darcy flow solver
problem = rok.DarcyProblem(mesh)
problem.setFluidDensity(rho)
problem.setFluidViscosity(mu)
problem.setRockPermeability(k)
problem.setSourceRate(f)
# problem.addPressureBC(5e6, 'left')
# problem.addPressureBC(5e5, 'right')
problem.addVelocityComponentBC(v_left, 'x', 'left')
problem.addVelocityComponentBC(v_bottom, 'y', 'bottom')
problem.addVelocityComponentBC(v_right, 'x', 'right')
problem.addVelocityComponentBC(v_top, 'y', 'top')
# problem.addVelocityComponentBC(rok.Constant(0.0), 'y', 'bottom')
# problem.addVelocityComponentBC(rok.Constant(0.0), 'y', 'top')

flow = rok.DarcySolver(problem, method=method)
flow.solve()

rok.File(resultsdir + 'flow.pvd').write(flow.u, flow.p, k)

rok.plot(k)
plt.axis('off')
plt.show()

rok.plot(flow.u)
plt.axis('off')
plt.show()

rok.plot(flow.p)
plt.axis('off')
plt.show()
