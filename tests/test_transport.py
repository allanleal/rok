import firedrake as fire
import numpy as np
import rok


def test_transport_step(num_regression):

    nx, ny = 5, 5  # number of cells along x and y
    nsteps = 10  # number of time steps

    velocity = fire.Constant([1.0e-2, 0.0])
    diffusion = fire.Constant(1.0e-3)
    source = fire.Constant(0.0)
    dt = 1.0

    mesh = fire.UnitSquareMesh(nx, ny, quadrilateral=True)

    V = fire.FunctionSpace(mesh, "CG", 1)

    bc = fire.DirichletBC(V, fire.Constant(1.0), 1)

    u = fire.Function(V)

    transport = rok.TransportSolver()
    transport.setVelocity(velocity)
    transport.setDiffusion(diffusion)
    transport.setSource(source)
    transport.setBoundaryConditions([bc])

    step = 0

    data = []
    data.append(u.dat.data)

    while step < nsteps:
        transport.step(u, dt)
        step += 1
        data.append(u.dat.data)

    data = {"u(step={})".format(i): u for i, u in enumerate(data)}

    num_regression.check(data)
