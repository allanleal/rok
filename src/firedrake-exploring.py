from firedrake import *
from numpy import *
from reaktoro import *


mesh = UnitSquareMesh(5, 5, quadrilateral=True)

V = FunctionSpace(mesh, 'CG', 1)

print(V.cell_node_map())

# f = Function(V)

# bc = DirichletBC(V, 1.0, 1)

# bc.apply(f)

# print(where(f.dat.data == 1.0))



# DirichletElementAmountBC(V, state, ielement, )

# DirichletSpeciesAmountBC

# phases = Phases(...)
# reactions = Reactions(phases)

# system = ChemicalSystem(phases, reactions, partition)

# partition.addInertReaction('Fe+2 = Fe+3 + e-')



# transport = ChemicalTransportSolver(mesh, system)
# transport.step(states, dt)
