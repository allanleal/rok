import firedrake as fire
import reaktoro as rkt
import rok

# Auxiliary time related constants
second = 1
minute = 60
hour = 60 * minute
day = 24 * hour
week = 7 * day
year = 365 * day

# Parameters for the reactive transport simulation
nx = 25           # the number of mesh cells along the x-coordinate
ny = 25           # the number of mesh cells along the y-coordinate
nz = 25           # the number of mesh cells along the y-coordinate

nsteps = 100       # the number of time steps

D  = fire.Constant(1.0e-9)                # the diffusion coefficient (in units of m2/s)
v  = fire.Constant([1.0/week, 0.0])  # the fluid pore velocity (in units of m/s)
dt = 30*minute                           # the time step (in units of s)
T  = 60.0 + 273.15                        # the temperature (in units of K)
P  = 100 * 1e5                            # the pressure (in units of Pa)

# Initialise the mesh
# mesh = fire.UnitIntervalMesh(nx)
# mesh = fire.UnitCubeMesh(nx, ny, nz)
mesh = fire.UnitSquareMesh(nx, ny, quadrilateral=True)

V = fire.FunctionSpace(mesh, 'CG', 1)

# Initialise the database
database = rkt.Database('supcrt98.xml')

# Initialise the chemical editor
editor = rkt.ChemicalEditor(database)
editor.addAqueousPhase('H2O(l) H+ OH- Na+ Cl- Ca++ Mg++ HCO3- CO2(aq) CO3--')
editor.addMineralPhase('Quartz')
editor.addMineralPhase('Calcite')
editor.addMineralPhase('Dolomite')

# Initialise the chemical system
system = rkt.ChemicalSystem(editor)

# Define the initial condition of the reactive transport modeling problem
problem_ic = rkt.EquilibriumProblem(system)
problem_ic.setTemperature(T)
problem_ic.setPressure(P)
problem_ic.add('H2O', 1.0, 'kg')
problem_ic.add('NaCl', 0.7, 'mol')
problem_ic.add('CaCO3', 10, 'mol')
problem_ic.add('SiO2', 10, 'mol')

# Define the boundary condition of the reactive transport modeling problem
problem_bc = rkt.EquilibriumProblem(system)
problem_bc.setTemperature(T)
problem_bc.setPressure(P)
problem_bc.add('H2O', 1.0, 'kg')
problem_bc.add('NaCl', 0.90, 'mol')
problem_bc.add('MgCl2', 0.05, 'mol')
problem_bc.add('CaCl2', 0.01, 'mol')
problem_bc.add('CO2', 0.75, 'mol')

# Calculate the equilibrium states for the initial and boundary conditions
state_ic = rkt.equilibrate(problem_ic)
state_bc = rkt.equilibrate(problem_bc)

# Scale the volumes of the phases in the initial condition such that their sum is 1 m3
state_ic.scalePhaseVolume('Aqueous', 0.1, 'm3')
state_ic.scalePhaseVolume('Quartz', 0.882, 'm3')
state_ic.scalePhaseVolume('Calcite', 0.018, 'm3')

# Scale the volume of the boundary equilibrium state to 1 m3
state_bc.scaleVolume(1.0)

# Initialise the chemical field
field = rok.ChemicalField(system, V)
field.fill(state_ic)

# Initialize the chemical transport solver
transport = rok.ChemicalTransportSolver(field)
transport.addBoundaryCondition(state_bc, 1)  # 1 means left side in a rectangular mesh
transport.setVelocity([v])
transport.setDiffusion([D])

out_species = ['Ca++', 'Mg++', 'Calcite', 'Dolomite', 'CO2(aq)', 'HCO3-', 'Cl-', 'H2O(l)']
out_elements = ['H', 'O', 'C', 'Ca', 'Mg', 'Na', 'Cl']

nout = [fire.Function(V, name=name) for name in out_species]
bout = [fire.Function(V, name=name) for name in out_elements]


# Create the output file
file_species_amounts = fire.File('results/demo-chemicaltransport/species-amounts.pvd')
file_element_amounts = fire.File('results/demo-chemicaltransport/element-amounts.pvd')
file_porosity = fire.File('results/demo-chemicaltransport/porosity.pvd')

t = 0.0
step = 0

while step <= nsteps:
    print('Time: {:<5.2f} day ({}/{})'.format(t/day, step, nsteps))

    # For each selected species, output its molar amounts
    for f in nout:
        f.assign(field.speciesAmount(f.name()))

    # For each selected species, output its molar amounts
    for f in bout:
        f.assign(field.elementAmountInPhase(f.name(), 'Aqueous'))


    file_species_amounts.write(*nout)
    file_element_amounts.write(*bout)
    file_porosity.write(field.porosity())

    # Perform one transport step from `t` to `t + dt`
    transport.step(field, dt)

    # Update the current time
    step += 1
    t += dt

