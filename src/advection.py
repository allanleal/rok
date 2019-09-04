from firedrake import *
import math

mesh = UnitSquareMesh(40, 40, quadrilateral=True)

V = FunctionSpace(mesh, "DQ", 1)
W = VectorFunctionSpace(mesh, "CG", 1)

v = Function(W).interpolate(Constant((1.0e-2, 0.0)))

T = 100
dt = 1.0
dtc = Constant(dt)
u_in = Constant(1.0)

du = TrialFunction(V)
w = TestFunction(V)

a = (w * du)*dx

# Define normal velocity field vn
n = FacetNormal(mesh)
vn = 0.5*(dot(v, n) + abs(dot(v, n)))

u = Function(V)
u1 = Function(V)
u2 = Function(V)
u3 = Function(V)

L1 = dtc*(dot(grad(w), u*v)*dx
    # - conditional(dot(v, n) < 0, w*dot(v, n)*u_in, 0.0)*ds    # inflow condition
    - jump(w)*jump(vn*u)*dS # flow across internal faces
    - conditional(dot(v, n) > 0, w*dot(v, n)*u, 0.0)*ds       # outflow condition
    - w*dot(v, n)*u_in*ds(1)    # inflow condition
)

L2 = replace(L1, {u: u1})
L3 = replace(L1, {u: u2})

du = Function(V)

params = {'ksp_type': 'preonly', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}
prob1 = LinearVariationalProblem(a, L1, du)
solv1 = LinearVariationalSolver(prob1, solver_parameters=params)
prob2 = LinearVariationalProblem(a, L2, du)
solv2 = LinearVariationalSolver(prob2, solver_parameters=params)
prob3 = LinearVariationalProblem(a, L3, du)
solv3 = LinearVariationalSolver(prob3, solver_parameters=params)

t = 0.0
step = 0

outfile = File("u.pvd")
outfile.write(u)

while t < T:
    solv1.solve()
    u1.assign(u + du)

    solv2.solve()
    u2.assign(0.75*u + 0.25*(u1 + du))

    solv3.solve()
    u.assign((1.0/3.0)*u + (2.0/3.0)*(u2 + du))

    step += 1
    t += dt

    umin = min(u.dat.data)
    umax = max(u.dat.data)

    if step % 20 == 0:
      outfile.write(u)
      print("t = {:<15.3f} step = {:<15} umin = {:<15.3f} umax = {:<15.3f}".format(t, step, umin, umax))

