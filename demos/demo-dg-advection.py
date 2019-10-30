import matplotlib.pyplot as plt
from firedrake import *

run_as_quad_mesh = True

# Parameters
Pe = Constant(1e12)
t_end = 10
dt = 0.1

# Create mesh and define function space
mesh = UnitSquareMesh(40, 40, quadrilateral=run_as_quad_mesh)

# Define function spaces
V = FunctionSpace(mesh, "DG", 1)
W = VectorFunctionSpace(mesh, "CG", 2)
x, y = SpatialCoordinate(mesh)

velocity = as_vector((0.5 - y, x - 0.5))
b = Function(W).interpolate(velocity)

bc = DirichletBC(V, Constant(0.0), "on_boundary", method="geometric")

# Define unknown and test function(s)
v = TestFunction(V)
u = TrialFunction(V)

# Initial condition
u0 = Function(V).interpolate(
    conditional(lt((x - 0.3) ** 2.0 + (y - 0.3) ** 2.0, 0.2 * 0.2), 1.0, 0.0)
)

# STABILIZATION
h = CellDiameter(mesh)
n = FacetNormal(mesh)
alpha = Constant(1e0)

theta = Constant(1.0)

# ( dot(v, n) + |dot(v, n)| )/2.0
bn = (dot(b, n) + abs(dot(b, n))) / 2.0


def a(u, v):
    # Bilinear form
    a_int = dot(grad(v), (1.0 / Pe) * grad(u) - b * u) * dx

    a_fac = (
        (1.0 / Pe) * (alpha / avg(h)) * dot(jump(u, n), jump(v, n)) * dS
        - (1.0 / Pe) * dot(avg(grad(u)), jump(v, n)) * dS
        - (1.0 / Pe) * dot(jump(u, n), avg(grad(v))) * dS
    )

    a_vel = dot(jump(v), bn("+") * u("+") - bn("-") * u("-")) * dS + dot(v, bn * u) * ds

    a = a_int + a_fac + a_vel
    return a


# Define variational forms
a0 = a(u0, v)
a1 = a(u, v)

A = (1 / dt) * inner(u, v) * dx - (1 / dt) * inner(u0, v) * dx + theta * a1 + (1 - theta) * a0

F = A

# Create files for storing results
ufile = File("results_demo-DG/u.pvd")

u = Function(V)
problem = LinearVariationalProblem(lhs(F), rhs(F), u, bcs=bc)
solver = LinearVariationalSolver(problem)

u.assign(u0)

# Time-stepping
t = 0.0

ufile.write(u)

while t < t_end:
    print(f"t = {t} \t end t = {t_end}")

    # Compute
    solver.solve()

    # Save to file
    ufile.write(u)

    # Move to next time step
    u0.assign(u)
    t += dt
