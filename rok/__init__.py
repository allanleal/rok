from firedrake import *
from reaktoro import *

del globals()[
    "ChemicalField"
]  # Reaktoro also has a ChemicalField class, different from the one in Rok

from .flow import DarcyProblem, DarcySolver
from .transport import TransportSolver
from .chemicaltransport import ChemicalDirichletBC, ChemicalTransportSolver
from .chemicalfield import ChemicalField
from .permeability import permeability
from .utils import DirichletExpressionBC
