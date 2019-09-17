import firedrake as fire
import reaktoro as rkt
import numpy as np
import rok
from sys import exit

import matplotlib.pyplot as plt
from gstools import SRF, Gaussian
from scipy.interpolate import interp2d

x = y = range(100)

model = Gaussian(dim=2, var=1, len_scale=10)
srf = SRF(model, seed=20170519)

test = srf([x, y], mesh_type='structured')
test *= 1e-12
test -= np.min(test)
test += 1e-12

print("min(test) = ", np.min(test))
print("max(test) = ", np.max(test))

# field = srf.structured([x, y])
# plt.imshow(test)
# plt.show()
x = np.arange(0.0, 1.0, 0.01)
y = np.arange(0.0, 1.0, 0.01)
xx, yy = np.meshgrid(x, y)
z = test

f = interp2d(x, y, z, kind='linear')

xnew = np.arange(0.0, 1.0, 0.001)
ynew = np.arange(0.0, 1.0, 0.001)
znew = f(xnew, ynew)

# plt.imshow(znew)
# plt.show()




# Auxiliary time related constants
second = 1
minute = 60
hour = 60 * minute
day = 24 * hour
week = 7 * day
year = 365 * day

# Parameters for the reactive transport simulation
nx = 100           # the number of mesh cells along the x-coordinate
ny = 100           # the number of mesh cells along the y-coordinate
nz = 25           # the number of mesh cells along the y-coordinate

# Initialise the mesh
mesh = rok.UnitSquareMesh(nx, ny, quadrilateral=True)
# mesh = rok.RectangleMesh(nx, 1, 1.0, 1/nx, quadrilateral=True)

x, y = rok.SpatialCoordinate(mesh)

# Model parameters
V = rok.FunctionSpace(mesh, "CG", 1)
V0 = rok.FunctionSpace(mesh, "DG", 0)

# Now make the VectorFunctionSpace corresponding to V.
W = rok.VectorFunctionSpace(mesh, V0.ufl_element())

# Next, interpolate the coordinates onto the nodes of W.
X = rok.interpolate(mesh.coordinates, W)

# X = rok.interpolate(mesh.coordinates, V0)



# k = rok.Function(V, name='Permeability')
k = rok.Function(V0, name='Permeability')

k0 = rok.Constant(1e-12)
gamma = 5.0
Nx, Ny = 10, 10
pi = 3.14159265359
# k.interpolate(k0*rok.exp(gamma*rok.cos(2*pi*Nx*(x-0.5))*rok.cos(2*pi*Ny*(y-0.5))))
# k.dat.data[:] = np.random.rand(len(k.dat.data)) * 1e-12

def mydata(xy):
    return f(xy[:, 0], xy[:, 1])

# kdata = np.copy(np.transpose(test))
# k.dat.data[:] = test.reshape(k.dat.data.shape)

print(X.dat.data_ro.shape)

kdata = [f(xy[0], xy[1]) for xy in X.dat.data_ro]
# k.dat.data[:] = mydata(X.dat.data_ro)
k.dat.data[:] = kdata


# k.dat.data[:] = 1e-12

# k.interpolate(rok.conditional(y >= 0.5, rok.Constant(1e-12), rok.Constant(1e-13)))

rok.File('results/demo-transport-in-heterogeneous-medium2/permeability.pvd').write(k)



# rho = rok.Constant(1000.0)  # water density (in units of kg/m3)


# TEMPORARY
rho = rok.Constant(1.0)  # water density (in units of kg/m3)




mu = rok.Constant(8.9e-4)  # water viscosity (in units of Pa*s)
# k = rok.Constant(1e-12)  # rock permeability (in units of m2)
f = rok.Constant(0.0)  # the source rate in the flow calculation


nsteps = 10000       # the number of time steps

D  = fire.Constant(1.0e-9)               # the diffusion coefficient (in units of m2/s)
v  = fire.Constant([1.0/week, 0.0])      # the fluid pore velocity (in units of m/s)
# dt = 30*minute                           # the time step (in units of s)

# Initialize the Darcy flow solver
problem = rok.DarcyProblem(mesh)
problem.setFluidDensity(rho)
problem.setFluidViscosity(mu)
problem.setRockPermeability(k)
problem.setSourceRate(f)
problem.addPressureBC(1e5, 'right')
problem.addPressureBC(1e5 + 1000, 'left')
# problem.addVelocityBC(v, 'left')
problem.addVelocityComponentBC(rok.Constant(0.0), 'y', 'bottom')
problem.addVelocityComponentBC(rok.Constant(0.0), 'y', 'top')

flow = rok.DarcySolver(problem)
flow.solve()

flowfile = rok.File('results/demo-transport-in-heterogeneous-medium2/flow.pvd')
flowfile.write(flow.u, flow.p, k)

# exit()

# Initialize the transport solver
transport = rok.TransportSolver()
transport.setVelocity(flow.u)
transport.setDiffusion(D)
transport.setSource(f)

bc = fire.DirichletBC(V, fire.Constant(1.0), 1)
transport.setBoundaryConditions([bc])

t = 0.0
step = 0

c = fire.Function(V)

file = rok.File('results/demo-transport-in-heterogeneous-medium2/concentration.pvd')
file.write(c)

max_ux = np.max(flow.u.dat.data[:,0])
max_uy = np.max(flow.u.dat.data[:,1])
delta_x = 1/nx
# cfl = 1.0
cfl = 0.3
# cfl = v*dt/dx
dt = cfl/max(max_ux, max_uy) * delta_x

print('max(u) = ', np.max(flow.u.dat.data[:,0]))
print('max(k) = ', np.max(k.dat.data))

print('div(u)*dx =', rok.assemble(rok.div(flow.u)*rok.dx))
print('dt = {} minute'.format(dt/minute))

# exit()

while step <= nsteps:
    print('Time: {:<5.2f} day ({}/{})'.format(t/day, step, nsteps))

    transport.step(c, dt)

    t += dt
    step += 1

    # sumc = rok.assemble(c*rok.dx)

    # print('Sum[c] =', sumc)

    if step % 20 == 0:
        file.write(c)

