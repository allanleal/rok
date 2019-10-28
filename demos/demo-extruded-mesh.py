from firedrake import *
import matplotlib.pyplot as plt

Lx, Ly, Lz = 1, 1, 1
nx, ny, nz = 50, 50, 10
base_mesh = RectangleMesh(nx, ny, Lx, Ly, quadrilateral=True)
mesh = ExtrudedMesh(base_mesh, nz, 1 / nz)

V = FunctionSpace(mesh, "CG", 1)
u = TrialFunction(V)
v = TestFunction(V)

k = Constant(1)
f = Constant(0)

x, y, _ = SpatialCoordinate(mesh)
circle_source = conditional((x - 0.5) ** 2 + (y - 0.5) ** 2 <= 0.0025, -x ** 2 + x - y ** 2 + y, 0)
bc_bottom = DirichletBC(V, circle_source, "bottom")

a = dot(k * grad(u), grad(v)) * dx
L = f * v * dx

u = Function(V)
problem = LinearVariationalProblem(a, L, u, bcs=bc_bottom)
solver = LinearVariationalSolver(problem)

solver.solve()

output = File("results/ext_mesh.pvd")
output.write(u)
