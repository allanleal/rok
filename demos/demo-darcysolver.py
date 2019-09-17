import rok


rho = rok.Constant(1000.0)  # water density (in units of kg/m3)
mu = rok.Constant(8.9e-4)  # water viscosity (in units of Pa*s)
k = rok.Constant(1e-12)  # rock permeability (in units of m2)
f = rok.Constant(0.0)

mesh = rok.UnitSquareMesh(32, 32, quadrilateral=True)

problem = rok.DarcyProblem(mesh)
problem.setFluidDensity(rho)
problem.setFluidViscosity(mu)
problem.setRockPermeability(k)
problem.setSourceRate(f)
problem.addPressureBC(1e5, 'right')
problem.addVelocityBC(rok.Constant([1e-5, 0.0]), 'left')
problem.addVelocityComponentBC(0.0, 'y', 'left')
problem.addVelocityComponentBC(0.0, 'y', 'bottom')
problem.addVelocityComponentBC(0.0, 'y', 'top')

solver = rok.DarcySolver(problem)
solver.solve()

file = rok.File('demo-darcysolver.pvd')

file.write(solver.u, solver.p)
