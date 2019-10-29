from firedrake import *
from reaktoro import *

del globals()[
    "ChemicalField"
]  # Reaktoro also has a ChemicalField class, different from the one in Rok

from .flow import DarcyProblem, DarcySolver
from .transport import TransportSolver
from .chemicaltransport import ChemicalDirichletBC, ChemicalTransportSolver
from .chemicalfield import ChemicalField
from .random_field import random_field_generator
from .permeability import permeability
from .porosity import porosity, rough_porosity
from .utils import DirichletExpressionBC
