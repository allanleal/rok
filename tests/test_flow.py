import rok


def test_DarcyVelocityBC():
    mesh = rok.UnitSquareMesh(500, 500, quadrilateral=True)
    V = rok.VectorFunctionSpace(mesh, 'CG', 1)
    U = rok.FunctionSpace(mesh, 'CG', 1)
    x, y = rok.SpatialCoordinate(mesh)

    rho = rok.Function(U)
    rho.assign(rok.Constant(2.0))

    ubc = rok.DarcyVelocityBC(V, rok.Constant([6.0, -3.0]), 1)

    ubc.update(rho)

    boundarydofs = V.boundary_nodes(1, "topological")

    assert (ubc.qboundary.dat.data[boundarydofs, 0] == 12.0).all()
    assert (ubc.qboundary.dat.data[boundarydofs, 1] == -6.0).all()


def test_DirichletExpressionBC():
    mesh = rok.UnitSquareMesh(500, 500, quadrilateral=True)
    V = rok.VectorFunctionSpace(mesh, 'CG', 1)
    U = rok.FunctionSpace(mesh, 'CG', 1)
    x, y = rok.SpatialCoordinate(mesh)

    rho = rok.Function(U)

    ubc = rok.DirichletExpressionBC(V, rho * rok.Constant([6.0, -3.0]), 1)

    rho.assign(rok.Constant(2.0))

    ubc.update()

    boundarydofs = V.boundary_nodes(1, "topological")

    assert (ubc.uboundary.dat.data[boundarydofs, 0] == 12.0).all()
    assert (ubc.uboundary.dat.data[boundarydofs, 1] == -6.0).all()

# test_DarcyVelocityBC()
test_DirichletExpressionBC()
