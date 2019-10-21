import numpy as np
import rok
import time

# Time constants
second = 1
minute = 60
hour = 60 * minute
day = 24 * hour
week = 7 * day
year = 365 * day

# Mathematical constants
pi = 3.14159265359

# Parameters for the reactive transport simulation
nx = 100               # the number of mesh cells along the x-axis
ny = 100               # the number of mesh cells along the y-axis
nz = 25                # the number of mesh cells along the y-axis
Lx = 1.6               # the length of the mesh along the x-axis
Ly = 1.0               # the length of the mesh along the y-axis
nsteps = 100           # the number of time steps
cfl = 0.5              # the CFL number to be used in the calculation of time step
T  = 60.0 + 273.15     # the temperature (in units of K)
P  = 1e5               # the pressure (in units of Pa)
tend = 1*day           # the final time (in units of s)

# method_flow = 'cgls'
# method_flow = 'dgls'
method_flow = 'sdhm'

method_transport = 'supg'

# The path to where the result files are output
resultsdir = f'results/demo-reactiveflow/mesh-{nx}x{ny}-cfl-{cfl}-flow-{method_flow}-transport-{method_transport}/'

# Initialise the mesh
mesh = rok.RectangleMesh(nx, ny, Lx, Ly, quadrilateral=True)

V = rok.FunctionSpace(mesh, "CG", 1)

x, y = rok.SpatialCoordinate(mesh)

# Model parameters
rho = rok.Constant(1000.0)     # the water density (in units of kg/m3)
mu = rok.Constant(8.9e-4)      # the water viscosity (in units of Pa*s)
k = rok.permeability(V)        # the heterogeneous rock permeability
f = rok.Constant(0.0)          # the source rate in the flow calculation
D  = rok.Constant(1.0e-9)      # the diffusion coefficient (in units of m2/s)

# Initialise the database
database = rok.Database('supcrt98.xml')

# Initialise the chemical editor
editor = rok.ChemicalEditor(database)
editor.addAqueousPhase('H2O(l) H+ OH- Na+ Cl- Ca++ Mg++ HCO3- CO2(aq) CO3--')
editor.addMineralPhase('Quartz')
editor.addMineralPhase('Calcite')
editor.addMineralPhase('Dolomite')

# Initialise the chemical system
system = rok.ChemicalSystem(editor)

# Define the initial condition of the reactive transport modeling problem
problem_ic = rok.EquilibriumProblem(system)
problem_ic.setTemperature(T)
problem_ic.setPressure(P)
problem_ic.add('H2O', 1.0, 'kg')
problem_ic.add('NaCl', 0.7, 'mol')
problem_ic.add('CaCO3', 10, 'mol')
problem_ic.add('SiO2', 10, 'mol')

# Define the boundary condition of the reactive transport modeling problem
problem_bc = rok.EquilibriumProblem(system)
problem_bc.setTemperature(T)
problem_bc.setPressure(P)
problem_bc.add('H2O', 1.0, 'kg')
problem_bc.add('NaCl', 0.90, 'mol')
problem_bc.add('MgCl2', 0.05, 'mol')
problem_bc.add('CaCl2', 0.01, 'mol')
problem_bc.add('CO2', 0.75, 'mol')

# Calculate the equilibrium states for the initial and boundary conditions
state_ic = rok.equilibrate(problem_ic)
state_bc = rok.equilibrate(problem_bc)

# Scale the volumes of the phases in the initial condition such that their sum is 1 m3
state_ic.scalePhaseVolume('Aqueous', 0.1, 'm3')
state_ic.scalePhaseVolume('Quartz', 0.882, 'm3')
state_ic.scalePhaseVolume('Calcite', 0.018, 'm3')

# Scale the volume of the boundary equilibrium state to 1 m3
state_bc.scaleVolume(1.0)

# Initialise the chemical field
field = rok.ChemicalField(system, V)
field.fill(state_ic)
field.update()

# Initialize the Darcy flow solver
problem = rok.DarcyProblem(mesh)
problem.setFluidDensity(rho)
problem.setFluidViscosity(mu)
problem.setRockPermeability(k)
problem.setSourceRate(f)
problem.addPressureBC(1e5 + 1000, 'left')
problem.addPressureBC(1e5, 'right')
problem.addVelocityComponentBC(rok.Constant(0.0), 'y', 'bottom')
problem.addVelocityComponentBC(rok.Constant(0.0), 'y', 'top')

flow = rok.DarcySolver(problem, method=method_flow)

rok.File(resultsdir + 'flow.pvd').write(flow.u, flow.p, k)

# Initialize the chemical transport solver
transport = rok.ChemicalTransportSolver(field, method=method_transport)
transport.addBoundaryCondition(state_bc, 1)  # 1 means left side in a rectangular mesh
transport.setVelocity([flow.u])
transport.setDiffusion([D])

out_species = ['Ca++', 'Mg++', 'Calcite', 'Dolomite', 'CO2(aq)', 'HCO3-', 'Cl-', 'H2O(l)']
out_elements = ['H', 'O', 'C', 'Ca', 'Mg', 'Na', 'Cl']

nout = [rok.Function(V, name=name) for name in out_species]
bout = [rok.Function(V, name=name) for name in out_elements]

# Create the output file
file_species_amounts = rok.File(resultsdir + 'species-amounts.pvd')
file_element_amounts = rok.File(resultsdir + 'element-amounts.pvd')
file_porosity = rok.File(resultsdir + 'porosity.pvd')
file_volume = rok.File(resultsdir + 'volume.pvd')
file_ph = rok.File(resultsdir + 'ph.pvd')

t = 0.0
step = 0

# rho.assign(field.densities()[0])

flow.solve()

rok.File(resultsdir + 'flow.pvd').write(flow.u, flow.p, k)

# Set the pressure field to the chemical field
field.setPressures(flow.p.dat.data)


max_ux = np.max(flow.u.dat.data[:,0])
max_uy = np.max(flow.u.dat.data[:,1])
delta_x = Lx/nx
delta_y = Ly/nx

dt = cfl/max(max_ux/delta_x, max_uy/delta_y)

print('max(u) = ', np.max(flow.u.dat.data[:,0]))
print('max(k) = ', np.max(k.dat.data))
print('div(u)*dx =', rok.assemble(rok.div(flow.u)*rok.dx))
print('dt = {} minute'.format(dt/minute))

start_time = time.time()

while t < tend and step < nsteps:
    elapsed_time = (time.time() - start_time)/hour
    final_time = elapsed_time*(tend/t - 1) if t != 0.0 else 0.0

    print('Progress at step {}: {:.2f} hour ({:.2f}% of {:.2f} days), elapsed time is {:.2f} hour (estimated to end in {:.2f} hours)'.format(step, t/hour, t/tend*100, tend/day, elapsed_time, final_time))

    if step % 10 == 0:
        # For each selected species, output its molar amounts
        for f in nout:
            f.assign(field.speciesAmount(f.name()))

        # For each selected species, output its molar amounts
        for f in bout:
            f.assign(field.elementAmountInPhase(f.name(), 'Aqueous'))

        file_species_amounts.write(*nout)
        file_element_amounts.write(*bout)
        file_porosity.write(field.porosity())
        file_volume.write(field.volume())
        file_ph.write(field.pH())

    # Perform one transport step from `t` to `t + dt`
    transport.step(field, dt)

    # rho.assign(field.densities()[0])

    # Update the current time
    t += dt
    step += 1

