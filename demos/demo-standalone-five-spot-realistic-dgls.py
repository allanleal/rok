import numpy as np
import rok
from firedrake import *


try:
    import matplotlib.pyplot as plt
    plt.rcParams['contour.corner_mask'] = False
    plt.close('all')
except:
    warning("Matplotlib not imported")

resultsdir = f'results/demo-standalone-five-spot/'

nx, ny = 100, 100
Lx, Ly = 1e3, 1e3
quadrilateral = True
mesh = RectangleMesh(nx, ny, Lx, Ly, quadrilateral=quadrilateral)

plot(mesh)
plt.axis('off')

degree = 1
pressure_family = 'DG'
velocity_family = 'DG'
U = VectorFunctionSpace(mesh, velocity_family, degree)
V = FunctionSpace(mesh, pressure_family, degree)
W = U * V

# For some parameters
dG0 = FunctionSpace(mesh, 'DG', 0)

# Trial and test functions
u, p = TrialFunctions(W)
v, q = TestFunctions(W)
solution = Function(W)

# Mesh entities
n = FacetNormal(mesh)
h = CellDiameter(mesh)
x, y = SpatialCoordinate(mesh)


# Model parameters
k = rok.permeability(dG0, minval=1, maxval=1e3)
mu = Constant(1.0)
rho = Constant(0.0)
g = Constant((0.0, 0.0))

# Boundary conditions
h_well = Constant(Lx / nx)
# The following strong conditions could be used with caution. TODO: Actually it should be fixed.
# v_top = conditional(And(Lx - x <= h_well, x <= Lx), as_vector([0, 1]), as_vector([0, 0]))
# v_bottom = conditional(And(x <= h_well, x > 0), as_vector([0, 1]), as_vector([0, 0]))
# v_left = conditional(And(y <= h_well, y > 0), as_vector([1, 0]), as_vector([0, 0]))
# v_right = conditional(And(Ly - y <= h_well, y <= Ly), as_vector([1, 0]), as_vector([0, 0]))
# bc1 = DirichletBC(W[0], v_left, 1, method='geometric')
# bc2 = DirichletBC(W[0], v_right, 2, method='geometric')
# bc3 = DirichletBC(W[0], v_bottom, 3, method='geometric')
# bc4 = DirichletBC(W[0], v_top, 4, method='geometric')
# bcs = [bc1, bc2, bc3, bc4]
# bcs = [bc1, bc3]
# bcs = [bc2, bc4]
bcs = []

# Source term
f = Function(dG0)
f_cut = 1
f.interpolate(
    conditional(
        And(x <= f_cut * h_well, y <= f_cut * h_well),
        Constant(50.),
        conditional(
            And(Lx - x <= f_cut * h_well, Ly - y <= f_cut * h_well),
            Constant(-50.),
            Constant(0.0)
        )
    )
)

# Stabilizing parameters
# Some notes: when using proper values of numerical flux penalty terms eta_p and eta_u, there is no need to include
# a mesh dependent parameter multiplying the stabilizing terms. Actually, with a proper sufficient large choice of
# eta_p and eta_u, there is no need to add mass conservation residual, neither the curl of Darcy's law. Maybe such
# stabilizing mechanisms can have an important role in the convergence rate improvements. The Darcy's law Least-Squares
# residual is important and can not be neglected. Due to its inclusion in the form, there is no need to choose a proper
# pair of elements and spaces which are compatible according to Brezzi-Babuska lemma.
has_mesh_characteristic_length = False
delta_0 = Constant(1)
delta_1 = Constant(-1 / 2)
delta_2 = Constant(1 / 2)
delta_3 = Constant(0)
eta_p = Constant(100)
eta_u = Constant(100)
h_avg = (h('+') + h('-')) / 2.
if has_mesh_characteristic_length:
    delta_2 = delta_2 * h * h
    delta_3 = delta_3 * h * h

# Mixed classical terms
a = (dot((mu / k) * u, v) - div(v) * p - delta_0 * q * div(u)) * dx
L = -delta_0 * f * q * dx - dot(rho * g, v) * dx #- p_boundaries * dot(v, n) * (ds(1) + ds(2) + ds(3) + ds(4))

# DG terms
a += jump(v, n) * avg(p) * dS - \
     avg(q) * jump(u, n) * dS

# Edge stabilizing terms
a += (eta_u * h_avg) * avg(mu / k) * (jump(u, n) * jump(v, n)) * dS + \
     (eta_p / h_avg) * avg(k / mu) * dot(jump(q, n), jump(p, n)) * dS

# Stabilizing terms
a += delta_1 * inner((k / mu) * ((mu / k) * u + grad(p)), (mu / k) * v + grad(q)) * dx
a += delta_2 * (mu / k) * div(u) * div(v) * dx
a += delta_3 * inner((k / mu) * curl((mu / k) * u), curl((mu / k) * v)) * dx
L += delta_2 * (mu / k) * f * div(v) * dx

# Weakly imposed BC
a += dot(v, n) * p * ds - \
     q * dot(u, n) * ds
# L += -dot(v, n) * p_L * ds(1) - \
#      dot(v, n) * p_R * ds(2)
a += eta_u / h * inner(dot(v, n), dot(u, n)) * ds

F = a - L

solver_parameters = {
    "mat_type": "aij",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps"
}

solve(lhs(F) == rhs(F), solution, bcs=bcs, solver_parameters=solver_parameters)
u_h, p_h = solution.split()
flux_h = Function(W.sub(0), name='Mass flux')
flux_h.project(k / mu * u_h)
u_h.rename('Velocity', 'label')
p_h.rename('Pressure', 'label')
k.rename('Permeability', 'label')

plot(p_h)
plt.axis('off')
plt.show()

plot(u_h)
plt.axis('off')
plt.show()

plot(flux_h)
plt.axis('off')
plt.show()

plot(k)
plt.axis('off')
plt.show()

output = File(resultsdir + 'flow.pvd', project_output=True)
output.write(u_h, p_h, flux_h, k)

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
nsteps = 10000    # the number of time steps
cfl = 0.2        # the CFL number to be used in the calculation of time step

V_transport = rok.FunctionSpace(mesh, "DG", 1)

D = rok.Constant(1.0e-9)

c = rok.Function(V_transport, name="Concentration")

c_inj = rok.Constant(1.0)

# Source term
f_c = Function(dG0)
f_cut = 1
f_c.interpolate(
    conditional(
        And(x <= f_cut * h_well, y <= f_cut * h_well),
        c_inj * f,
        0
    )
)

# Initialize the transport solver
transport = rok.TransportSolver(method='dg')
transport.setVelocity(u_h)
transport.setDiffusion(D)
transport.setSource(f_c)

# bc = rok.DirichletBC(V_transport, cL, 1, method='geometric')
# transport.setBoundaryConditions([bc])

t = 0.0
step = 0

file = rok.File(resultsdir + 'concentration.pvd')
file.write(c)

max_ux = np.max(u_h.dat.data[:, 0])
max_uy = np.max(u_h.dat.data[:, 1])
delta_x = Lx/nx
delta_y = Ly/nx

dt = cfl/max(max_ux/delta_x, max_uy/delta_y)

print('max(u) = ', np.max(u_h.dat.data[:, 0]))
print('max(k) = ', np.max(k.dat.data))

print('div(u)*dx =', rok.assemble(rok.div(u_h)*rok.dx))
print('dt = {} minute'.format(dt/minute))

while step <= nsteps:
    print('Time: {:<5.2f} day ({}/{})'.format(t/day, step, nsteps))

    transport.step(c, dt)

    t += dt
    step += 1

    if step % 50 == 0:
        file.write(c)

