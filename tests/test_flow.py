import numpy as np
import rok


rho = rok.Constant(1000.0)  # water density (in units of kg/m3)
mu = rok.Constant(8.9e-4)  # water viscosity (in units of Pa*s)
k = rok.Constant(1e-12)  # rock permeability (in units of m2)
f = rok.Constant(0.0)


def test_darcysolver_problem1(num_regression):

    mesh = rok.UnitSquareMesh(5, 5, quadrilateral=True)

    problem = rok.DarcyProblem(mesh)
    problem.setFluidDensity(rho)
    problem.setFluidViscosity(mu)
    problem.setRockPermeability(k)
    problem.setSourceRate(f)
    problem.addPressureBC(100e5, "left")
    problem.addPressureBC(1e5, "right")
    problem.addVelocityComponentBC(0.0, "y", "bottom")
    problem.addVelocityComponentBC(0.0, "y", "top")

    solver = rok.DarcySolver(problem, method="cgls")
    solver.solve()

    solver.u.dat.data, solver.p.dat.data

    data = {}
    data["ux"] = solver.u.dat.data[:, 0]
    data["uy"] = solver.u.dat.data[:, 1]
    data["p"] = solver.p.dat.data

    num_regression.check(data)


def test_transport_step_dgls(num_regression):

    mesh = rok.UnitSquareMesh(5, 5, quadrilateral=True)

    problem = rok.DarcyProblem(mesh)
    problem.setFluidDensity(rho)
    problem.setFluidViscosity(mu)
    problem.setRockPermeability(k)
    problem.setSourceRate(f)
    problem.addPressureBC(100e5, "left")
    problem.addPressureBC(1e5, "right")
    problem.addVelocityComponentBC(0.0, "y", "bottom")
    problem.addVelocityComponentBC(0.0, "y", "top")

    solver = rok.DarcySolver(problem, method="dgls")
    solver.solve()

    solver.u.dat.data, solver.p.dat.data

    data = {}
    data["ux"] = solver.u.dat.data[:, 0]
    data["uy"] = solver.u.dat.data[:, 1]
    data["p"] = solver.p.dat.data

    num_regression.check(data)


def test_darcysolver_problem2(num_regression):

    mesh = rok.UnitSquareMesh(5, 5, quadrilateral=True)

    problem = rok.DarcyProblem(mesh)
    problem.setFluidDensity(rho)
    problem.setFluidViscosity(mu)
    problem.setRockPermeability(k)
    problem.setSourceRate(f)
    problem.addPressureBC(1e5, "right")
    problem.addVelocityBC([1e-5, 0.0], "left")
    problem.addVelocityComponentBC(0.0, "y", "bottom")
    problem.addVelocityComponentBC(0.0, "y", "top")

    solver = rok.DarcySolver(problem)
    solver.solve()

    solver.u.dat.data, solver.p.dat.data

    data = {}
    data["ux"] = solver.u.dat.data[:, 0]
    data["uy"] = solver.u.dat.data[:, 1]
    data["p"] = solver.p.dat.data

    num_regression.check(data)
