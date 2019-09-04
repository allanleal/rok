from firedrake import *

T = 1.0
dt = 1.0e-2
dtc = Constant(dt)

# Load mesh
mesh = UnitSquareMesh(10, 10, quadrilateral=True)

# Defining the function spaces
V_dg = FunctionSpace(mesh, "DQ", 1)
V_cg = FunctionSpace(mesh, "CG", 1)
V_u  = VectorFunctionSpace(mesh, "CG", 2)



x, y = SpatialCoordinate(mesh)

nu = 1.0e-2
# delta = 1.0e-3
delta = 0.0


def update_phi_exact(phi_exact, t):
    phi_exact.assign(interpolate(sin(2*pi*(x - t)), V_dg))


def update_f(f, t):
    _u   =  sin(2*pi*(x - t))
    _ut  = -2*pi*cos(2*pi*(x - t))
    _ux  =  2*pi*cos(2*pi*(x - t))
    _uxx = -4*pi*pi*sin(2*pi*(x - t))
    _v   = nu*(1 + x)
    _vx  = nu
    _D   = delta*(1 + x)
    _Dx  = delta
    f.interpolate(_ut + _ux*_v + _u*_vx - _Dx*_ux - _D*_uxx)


# Create velocity Function
u = Function(V_u).interpolate(as_vector((nu*(1 + x), 0.0)))

# Test and trial functions
v   = TestFunction(V_dg)
phi = TrialFunction(V_dg)

phi_exact = Function(V_dg)

phi0 = Function(V_dg)

# Diffusivity
kappa = Function(V_dg).interpolate(delta*(1 + x))

# Source term
# f = Constant(0.0)
f = Function(V_dg)

# Penalty term
alpha = Constant(5.0)

# Mesh-related functions
n = FacetNormal(mesh)
h = CellDiameter(mesh)

# ( dot(v, n) + |dot(v, n)| )/2.0
un = (dot(u, n) + abs(dot(u, n)))/2.0

# Bilinear form
a_int = (v*phi/dtc - dot(grad(v), u*phi - kappa*grad(phi)))*dx

a_fac = avg(kappa)*(alpha/h('+'))*dot(jump(v, n), jump(phi, n))*dS \
      - avg(kappa)*dot(avg(grad(v)), jump(phi, n))*dS \
      - avg(kappa)*dot(jump(v, n), avg(grad(phi)))*dS

a_vel = dot(jump(v), un('+')*phi('+') - un('-')*phi('-') )*dS + (v*un*phi)*ds

a = a_int + a_fac + a_vel

# Linear form
L = (f + phi0/dtc)*v*dx

# Set up boundary condition (apply strong BCs)
# bc = DirichletBC(V_dg, Constant(1.0), 1, "geometric")
bc = DirichletBC(V_dg, phi_exact, 1, "geometric")

# Solution function
# phi_h = Function(V_dg)
phi_h = phi0

file = File("temperature.pvd")
# file.write(phi_exact)

step = 0

t = 0.0

update_phi_exact(phi_exact, t)
update_f(f, t)

A = assemble(a, bcs=[bc])
b = assemble(L, bcs=[bc])

while t < T:

    # Assemble and apply boundary conditions
    # assemble(a, tensor=A, bcs=[bc])
    assemble(L, tensor=b, bcs=[bc])

    # Solve system
    solve(A, phi_h, b, )

    update_phi_exact(phi_exact, t)
    update_f(f, t)

    file.write(phi_h)
    # file.write(phi_exact)

    print(norm(phi_exact - phi_h))

    umin = min(phi_h.dat.data)
    umax = max(phi_h.dat.data)

    print("t = {:<15.3f} step = {:<15} umin = {:<15.3f} umax = {:<15.3f}".format(t, step, umin, umax))

    step += 1
    t += dt

