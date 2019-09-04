'''
Created on Jul 22, 2015

@author: allan
'''

from firedrake import *
from reaktoro.PyReaktoro import (
    Database,
    ChemicalEditor,
    ChemicalSystem,
    EquilibriumProblem,
    ChemicalState
)

from .chemicalfield import *
from .chemicaltransport import *

# Auxiliary time related constants
second = 1
minute = 60
hour = 60 * minute
day = 24 * hour
year = 365 * day

# Parameters for the reactive transport simulation
xl = 0.0          # the x-coordinate of the left boundary
xr = 1.0          # the x-coordinate of the right boundary
yb = 0.0          # the y-coordinate of the bottom boundary
yt = 0.01         # the y-coordinate of the top boundary
nsteps = 100      # the number of steps in the reactive transport simulation
ncells = 100      # the number of cells in the discretization

D  = 1.0e-9       # the diffusion coefficient (in units of m2/s)
v  = 1.0/week     # the fluid pore velocity (in units of m/s)
dt = 30*minute    # the time step (in units of s)
T = 60.0 + 273.15 # the temperature (in units of K)
P = 100 * 1e5     # the pressure (in units of Pa)

t = 0.0



# # The velocity (in units of m/s)
# v = Constant((1.15740741e-5, 0.0)) # equivalent to 1 m/day

# # The diffusion coefficient (in units of m2/s)
# D = Constant(1.0e-9)

# Initialise the mesh
mesh = RectangleMesh(xl, yb, xr, yt, ncells, 1)

V = FunctionSpace(mesh, 'CG', 1)

# Initialise the database
database = Database('supcrt98.xml')

# Initialise the chemical editor
editor = ChemicalEditor(database)
editor.addAqueousPhase('H2O(l) H+ OH- Na+ Cl- Ca++ Mg++ HCO3- CO2(aq) CO3--')
editor.addMineralPhase('Quartz')
editor.addMineralPhase('Calcite')
editor.addMineralPhase('Dolomite')

# Initialise the chemical system
system = ChemicalSystem(editor)

# Define the initial condition of the reactive transport modeling problem
problem_ic = EquilibriumProblem(system)
problem_ic.setTemperature(T)
problem_ic.setPressure(P)
problem_ic.add('H2O', 1.0, 'kg')
problem_ic.add('NaCl', 0.7, 'mol')
problem_ic.add('CaCO3', 10, 'mol')
problem_ic.add('SiO2', 10, 'mol')

# Define the boundary condition of the reactive transport modeling problem
problem_bc = EquilibriumProblem(system)
problem_bc.setTemperature(T)
problem_bc.setPressure(P)
problem_bc.add('H2O', 1.0, 'kg')
problem_bc.add('NaCl', 0.90, 'mol')
problem_bc.add('MgCl2', 0.05, 'mol')
problem_bc.add('CaCl2', 0.01, 'mol')
problem_bc.add('CO2', 0.75, 'mol')

# Calculate the equilibrium states for the initial and boundary conditions
state_ic = equilibrate(problem_ic)
state_bc = equilibrate(problem_bc)

# Scale the volumes of the phases in the initial condition such that their sum is 1 m3
state_ic.scalePhaseVolume('Aqueous', 0.1, 'm3')
state_ic.scalePhaseVolume('Quartz', 0.882, 'm3')
state_ic.scalePhaseVolume('Calcite', 0.018, 'm3')

# Scale the volume of the boundary equilibrium state to 1 m3
state_bc.scaleVolume(1.0)

# Initialise the chemical field
field = ChemicalField(system, mobility, V)
field.fill(state_ic)

# Initialize the chemical transport solver
transport = ChemicalTransportSolver(field)
transport.addBoundaryCondition(state_bc, InletBoundary())
transport.setVelocity([v])
transport.setDiffusion([D])

out_species = ['Ca++', 'Mg++', 'Calcite', 'Dolomite', 'CO2(aq)', 'HCO3-', 'Cl-', 'H2O(l)']
out_elements = ['H', 'O', 'C', 'Ca', 'Mg', 'Na', 'Cl']

# Define the name of the result file
filename = 'nosupg-result-100C-300bar-dt-%d-cells-%d.xdmf' % (dt, ncells)

# Create the output file
file = File('result.pvd')

t = 0.0
step = 0

while step <= nsteps:
    print('Time: {:<5.2f} day ({}/{})'.format(t/day, step, nsteps))

    # For each selected species, output its molar amounts
    for species in out_species:
        file.write(field.speciesAmount(species), t)

#     # For each selected element, output its molar amounts
#     for element in out_elements:
#         file << (field.elementAmount(element, 'Aqueous'), t)

    # result = transport.result

    # file << (result.equilibrium.iterations, t)
    # file << (result.equilibrium.seconds, t)
    # file << (field.porosity(), t)
    # file << (field.volume(), t)

#     file << (field.volume(), t)
#     file << (field.ph(), t)

    # Perform one transport step from `t` to `t + dt`
    transport.step(field, dt)

    # # For each selected element, output its molar amounts
    # for element in out_elements:
    #     file << (transport.elementAmountInPhase(element, 'Aqueous'), t)

    # Update the current time
    t += dt


# print 'Statistics:'
# print 'Total simulation time = ', time.time() - begin
# print 'Total time spent on transport calculations = ', result.time
# print 'Total time spent on equilibrium calculations = ', result.time_equilibrium, '({:.2%})'.format(result.time_equilibrium/result.time)