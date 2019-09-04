from firedrake import *


class AdvectionSolver:

    def __init__(self, mesh, params=None):
        if params == None:
            params = {
                'ksp_type': 'preonly',
                'pc_type': 'bjacobi',
                'sub_pc_type': 'ilu'
            }

        V = FunctionSpace(mesh, "DQ", 1)
        W = VectorFunctionSpace(mesh, "CG", 1)

        u = Function(V)
        v = Function(W)
        uin = Constant(0.0)
        dt = Constant(0.0)

        n = FacetNormal(mesh)

        vn = 0.5*(dot(v, n) + abs(dot(v, n)))

        w = TestFunction(V)
        du = TrialFunction(V)

        a = (w * du)*dx

        L1 = dt*(dot(grad(w), u*v)*dx
            - w*dot(v, n)*uin*ds(1)                              # inflow condition
            - conditional(dot(v, n) > 0, w*dot(v, n)*u, 0.0)*ds  # outflow condition
            - jump(w)*jump(vn*u)*dS)                             # flow across internal faces

        u1 = Function(V)
        u2 = Function(V)

        L2 = replace(L1, {u: u1})
        L3 = replace(L1, {u: u2})

        du = Function(V)

        problem1 = LinearVariationalProblem(a, L1, du)
        problem2 = LinearVariationalProblem(a, L2, du)
        problem3 = LinearVariationalProblem(a, L3, du)

        solver1 = LinearVariationalSolver(problem1, solver_parameters=params)
        solver2 = LinearVariationalSolver(problem2, solver_parameters=params)
        solver3 = LinearVariationalSolver(problem3, solver_parameters=params)

        self.u = u
        self.u1 = u1
        self.u2 = u2
        self.uin = uin
        self.v = v
        self.dt = dt
        self.du = du
        self.solver1 = solver1
        self.solver2 = solver2
        self.solver3 = solver3


    def step(self, u, uin, v, dt):
        self.u.interpolate(u)
        self.v.interpolate(v)
        self.uin.assign(uin)
        self.dt.assign(dt)

        self.solver1.solve()
        self.u1.assign(self.u + self.du)

        self.solver2.solve()
        self.u2.assign(0.75*self.u + 0.25*(self.u1 + self.du))

        self.solver3.solve()
        u.assign((1.0/3.0)*self.u + (2.0/3.0)*(self.u2 + self.du))



T = 100
dt = 1.0
vx, vy = 1e-2, 0.0
uin = 1.0

mesh = UnitSquareMesh(40, 40, quadrilateral=True)

# P1 = FunctionSpace(mesh, "CG", 1)
P1 = FunctionSpace(mesh, "DQ", 1)

u = Function(P1)
v = Constant((vx, vy))

advection = AdvectionSolver(mesh)

outfile = File("u.pvd")
outfile.write(u)

t = 0.0
step = 0

while t < T:
    advection.step(u, uin, v, dt)

    step += 1
    t += dt

    outfile.write(u)
    umin = min(u.dat.data)
    umax = max(u.dat.data)

    if step % 20 == 0:
        outfile.write(u)
        print("t = {:<15.3f} step = {:<15} umin = {:<15.3f} umax = {:<15.3f}".format(t, step, umin, umax))

