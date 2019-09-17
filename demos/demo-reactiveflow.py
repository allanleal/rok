import firedrake as fire
import reaktoro as rkt
import numpy as np
import rok
from sys import exit


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

# Initialise the mesh
mesh = rok.UnitSquareMesh(nx, ny, quadrilateral=True)
# mesh = rok.RectangleMesh(nx, 1, 1.0, 1/nx, quadrilateral=True)

x, y = rok.SpatialCoordinate(mesh)

# Model parameters
V = rok.FunctionSpace(mesh, "CG", 1)
V0 = rok.FunctionSpace(mesh, "DG", 0)
k = rok.Function(V, name='Permeability')
# k = rok.Function(V0, name='Permeability')
# k = rok.Constant(1e-12)
# rho = rok.Function(V, name='Density')



rho = rok.Constant(1000)





# k.interpolate( (1.0 + 10*x + 100*y)**2 * 1e-12 )

k0 = rok.Constant(1e-12)
# gamma = 1.0e-4
gamma = 1.0
Nx, Ny = 1, 1
pi = 3.14159265359
k.interpolate(k0*rok.exp(gamma*rok.cos(2*pi*Nx*(x-0.5))*rok.cos(2*pi*Ny*(y-0.5))))
# k.interpolate(k0)
# k.interpolate(rok.Constant(k0))

# k.dat.data[:] = np.random.rand(len(k.dat.data)) * 1e-12
# k.dat.data[:] = [7.28438e-13, 9.59306e-13, 5.05296e-13, 3.24571e-13, 5.01791e-14, 1.83526e-13, 4.91487e-13, 7.07187e-13, 2.22113e-13, 7.36612e-13, 8.27471e-13, 3.26516e-13, 2.19883e-13, 4.28887e-13, 6.17511e-13, 7.71233e-13, 7.7803e-13, 5.55711e-13, 1.25734e-13, 5.26546e-14, 1.52494e-13, 5.92309e-13, 7.31233e-13, 5.08727e-13, 3.31972e-15, 4.81905e-13, 5.39409e-13, 2.89847e-13, 4.37526e-14, 2.05795e-13, 8.90035e-13, 3.1206e-13, 8.4921e-13, 7.68328e-13, 1.94733e-13, 6.11021e-13, 8.59918e-13, 4.77015e-13, 3.52934e-13, 6.04735e-13, 3.40923e-13, 2.10968e-14, 3.44295e-13, 1.776e-13, 7.21818e-13, 8.56147e-13, 5.09717e-13, 3.39179e-13, 2.31596e-13, 4.26288e-13, 2.86825e-13, 5.44138e-13, 2.49465e-13, 4.10957e-13, 7.02305e-13, 2.45098e-13, 6.9e-13, 1.8691e-13, 1.86323e-13, 5.90313e-13, 1.79142e-13, 1.42944e-13, 5.13149e-13, 9.4561e-13, 4.65563e-13, 9.50473e-15, 7.48744e-13, 1.93244e-13, 9.04915e-13, 6.36821e-13, 3.99848e-13, 8.02452e-13, 6.38855e-13, 9.23506e-13, 6.9585e-13, 9.6075e-13, 2.60423e-13, 3.04827e-13, 3.52777e-13, 8.36727e-13, 7.51933e-13, 3.9181e-13, 7.11967e-13, 3.45314e-13, 9.38353e-13, 7.00145e-13, 6.27417e-13, 2.22433e-14, 1.3815e-13, 7.22071e-13, 7.88762e-13, 5.64845e-13, 8.96491e-15, 2.75586e-13, 4.9656e-13, 1.08398e-13, 4.25724e-13, 9.25863e-13, 1.34281e-13, 2.03605e-14, 7.22904e-14, 4.61162e-13, 8.96563e-13, 7.91077e-13, 8.63211e-13, 5.71787e-13, 3.31653e-13, 5.3961e-14, 6.24012e-13, 8.06621e-13, 8.69646e-13, 6.03019e-13, 2.24304e-13, 8.55035e-13, 9.34655e-13, 3.18691e-13, 4.65815e-13, 3.48106e-13, 2.32593e-13, 4.47867e-13, 3.99885e-13, 4.18038e-13, 5.61408e-13, 5.31904e-13, 2.85137e-13, 8.81126e-13, 2.90165e-13, 9.40986e-13, 2.72495e-13, 1.75546e-13, 3.82659e-13, 3.09373e-13, 5.2391e-13, 3.57731e-13, 8.9396e-13, 9.23376e-14, 6.37627e-13, 1.3931e-13, 8.80027e-13, 4.42119e-14, 7.39883e-14, 7.02255e-13, 9.96967e-13, 2.71275e-13, 9.05454e-13, 8.55104e-13, 1.76848e-13, 1.10294e-13, 4.31869e-13, 9.89158e-13, 8.61267e-13, 7.15785e-13, 3.08706e-14, 5.59335e-13, 5.88502e-13, 3.76965e-13, 9.91432e-13, 7.10721e-13, 3.70021e-13, 8.80734e-13, 5.83455e-13, 5.86116e-13, 9.72533e-13, 3.9815e-13, 1.19639e-13, 9.71188e-13, 8.03273e-13, 4.98859e-13, 6.95254e-13, 6.37379e-13, 6.01258e-14, 5.85465e-14, 1.41115e-13, 1.54321e-13, 6.95041e-13, 2.16975e-13, 3.22858e-13, 5.84406e-13, 6.62747e-13, 7.15859e-15, 6.4103e-13, 3.99278e-13, 2.09438e-13, 3.66943e-13, 5.52408e-13, 4.11506e-13, 5.5276e-14, 7.53066e-13, 6.0373e-13, 4.89924e-13, 6.89123e-14, 6.76237e-13, 2.26986e-13, 5.09258e-13, 6.49326e-13, 8.52842e-13, 8.40265e-13, 5.63454e-13, 5.18345e-13, 1.31274e-13, 7.82591e-13, 9.91781e-13, 1.87143e-13, 6.05419e-13, 4.229e-13, 1.48824e-13, 4.8705e-13, 5.85329e-13, 8.88641e-13, 2.09286e-14, 1.35939e-13, 3.6155e-13, 8.19377e-13, 6.79647e-13, 2.88616e-13, 9.54622e-13, 7.68474e-13, 8.37235e-13, 4.85913e-13, 6.81871e-13, 8.27202e-13, 6.61996e-13, 2.69403e-14, 8.97347e-13, 4.61597e-13, 7.44206e-14, 7.04786e-14, 5.88889e-13, 1.46641e-13, 3.81819e-13, 5.34477e-13, 3.83265e-13, 4.42207e-13, 9.5143e-13, 8.90246e-15, 8.19118e-13, 7.83522e-13, 7.93464e-13, 1.73789e-13, 4.80418e-13, 9.75385e-13, 7.29277e-13, 5.47558e-13, 2.73689e-13, 6.75063e-13, 2.80504e-13, 5.25352e-13, 5.86261e-13, 2.73447e-13, 8.22779e-13, 7.42832e-13, 2.84616e-13, 8.52198e-13, 4.37654e-13, 8.94961e-13, 9.35683e-13, 2.94122e-13, 1.71172e-13, 8.94746e-13, 8.54951e-13, 6.40305e-13, 5.51585e-14, 5.10798e-13, 3.94662e-13, 8.05715e-15, 9.34631e-13, 1.28113e-13, 2.96534e-13, 2.59022e-13, 7.42519e-13, 4.87846e-13, 7.78278e-13, 1.71916e-13, 6.04063e-13, 1.99766e-13, 7.93951e-13, 2.93118e-13, 5.49833e-13, 4.69794e-13, 5.84229e-13, 4.78225e-13, 1.17312e-13, 3.46818e-13, 8.56774e-13, 7.82342e-13, 9.67036e-13, 9.29592e-13, 6.47063e-14, 4.82266e-13, 1.37144e-13, 5.92343e-13, 9.35622e-13, 7.46998e-13, 4.0497e-13, 3.49797e-13, 8.49151e-13, 8.9049e-13, 1.3245e-13, 6.96714e-13, 4.76242e-13, 2.33064e-13, 3.51182e-13, 1.78979e-13, 7.8315e-13, 7.0113e-13, 9.84524e-13, 8.2922e-13, 2.06281e-13, 2.12302e-13, 2.38679e-13, 8.56707e-13, 3.5315e-13, 2.76309e-13, 5.54114e-13, 8.57648e-13, 4.86894e-13, 5.43874e-13, 2.44549e-13, 8.45703e-13, 8.22041e-13, 4.59894e-13, 3.55426e-13, 7.72044e-13, 6.92342e-14, 7.54691e-14, 4.90787e-13, 9.57657e-13, 1.56279e-13, 2.95548e-13, 1.6347e-13, 4.84297e-13, 4.18478e-13, 8.43204e-14, 3.86021e-13, 3.96884e-13, 8.8306e-13, 2.90461e-14, 7.21131e-13, 3.81131e-13, 7.48553e-13, 2.82275e-14, 5.64123e-13, 2.64451e-13, 1.09682e-13, 9.22218e-13, 1.92991e-13, 1.83088e-13, 5.04687e-13, 3.40854e-13, 7.51352e-13, 6.86875e-13, 3.49558e-13, 6.28704e-13, 1.71453e-13, 2.70926e-13, 5.6384e-14, 5.4094e-13, 9.92941e-13, 5.76593e-13, 7.43806e-13, 8.8801e-13, 6.75648e-13, 4.20616e-13, 9.43753e-13, 2.92513e-14, 2.74967e-13, 5.42243e-13, 6.64402e-14, 9.05697e-14, 8.96914e-13, 8.91019e-13, 9.83523e-13, 6.64135e-13, 9.00112e-13, 9.29822e-13, 9.62569e-13, 8.04423e-13, 4.47881e-13, 3.26052e-13, 7.35952e-13, 2.4343e-13, 4.44821e-13, 7.32819e-14, 9.63588e-13, 1.2914e-13, 5.48081e-13, 9.70086e-13, 7.41836e-13, 1.36622e-13, 2.64904e-14, 2.85111e-13, 3.89779e-13, 8.552e-13, 6.99477e-13, 7.44931e-13, 4.74575e-13, 8.71342e-13, 9.99346e-13, 4.66708e-13, 7.21397e-13, 3.4875e-13, 9.79678e-13, 9.21468e-15, 7.75001e-13, 6.07891e-13, 6.77807e-13, 1.97094e-13, 2.13835e-13, 1.77359e-13, 9.79312e-13, 2.19443e-13, 2.16493e-13, 9.23221e-13, 9.08037e-14, 2.34942e-13, 2.13886e-13, 9.03485e-13, 3.12649e-13, 2.89668e-13, 2.74925e-13, 3.76359e-13, 9.01679e-13, 1.06627e-14, 3.40115e-13, 4.88296e-13, 8.54319e-15, 2.92308e-13, 8.02841e-13, 7.48531e-13, 1.82055e-15, 5.40884e-13, 9.31145e-13, 6.91298e-13, 9.16985e-13, 8.29926e-13, 6.09899e-13, 7.15494e-13, 2.94855e-13, 6.59222e-13, 5.20643e-13, 1.66841e-13, 1.62261e-13, 5.5227e-13, 7.46633e-15, 8.42944e-14, 4.76476e-13, 9.51185e-13, 8.10585e-13, 5.27213e-13, 7.82975e-15, 6.82007e-13, 5.26527e-13, 4.87193e-13, 8.76841e-13, 7.59083e-13, 6.91081e-13, 8.3545e-13, 1.80542e-13, 5.64526e-13, 9.12634e-14, 6.17087e-13, 1.81447e-13, 5.66464e-13, 4.15888e-13, 4.19664e-14, 3.97699e-13, 8.38775e-15, 1.80355e-13, 9.21695e-13, 5.52061e-13, 4.71586e-14, 5.20069e-13, 3.19194e-13, 9.51355e-13, 6.88147e-13, 4.62028e-13, 2.26837e-13, 8.57705e-13, 5.78158e-14, 4.32757e-13, 1.28118e-13, 3.32658e-14, 9.79754e-13, 3.65134e-13, 2.3976e-14, 1.68716e-13, 9.22471e-13, 4.19123e-13, 6.72097e-13, 4.63022e-13, 3.48212e-13, 2.1652e-13, 7.03493e-14, 5.93693e-13, 1.948e-13, 8.88881e-13, 2.56148e-13, 4.04191e-13, 7.32966e-13, 2.3793e-13, 3.52926e-13, 7.22921e-13, 2.01618e-13, 2.91411e-13, 6.11035e-13, 6.17788e-13, 6.38366e-13, 3.45617e-13, 4.53841e-13, 4.25357e-13, 7.23483e-13, 2.68988e-13, 1.2996e-13, 1.77437e-13, 6.19909e-13, 3.12888e-13, 1.53827e-13, 2.74657e-14, 6.87668e-13, 2.96941e-14, 4.97848e-13, 2.8374e-13, 8.0695e-13, 4.58602e-13, 7.09405e-13, 4.05682e-13, 6.84422e-13, 7.90826e-14, 9.73102e-13, 8.02245e-13, 8.82771e-13, 3.14442e-13, 4.90696e-13, 3.76088e-13, 8.07777e-13, 2.21769e-13, 3.55967e-13, 5.40597e-13, 7.16815e-13, 2.95159e-13, 3.17567e-13, 7.40027e-13, 9.63255e-13, 1.65663e-13, 9.08978e-13, 5.56556e-13, 1.57636e-13, 7.46518e-13, 9.11096e-13, 9.81265e-13, 9.13789e-13, 2.87476e-13, 8.37237e-13, 7.96299e-13, 8.98723e-13, 3.82688e-13, 8.60951e-14, 4.55892e-13, 4.63368e-13, 4.13344e-13, 1.06216e-13, 3.45443e-13, 4.05757e-13, 6.00441e-13, 8.00312e-13, 8.07686e-13, 7.88589e-13, 2.46346e-13, 1.2235e-13, 9.85004e-13, 1.90112e-13, 1.93391e-13, 3.78916e-13, 4.28568e-13, 2.23035e-13, 5.51725e-13, 2.82699e-13, 4.54868e-13, 6.5186e-13, 2.77551e-13, 1.72051e-13, 9.65085e-14, 2.93287e-13, 4.32381e-13, 2.94057e-13, 8.5392e-13, 9.8144e-13, 9.63996e-13, 2.5925e-13, 8.66785e-13, 6.29323e-13, 1.82654e-13, 5.13603e-13, 3.36377e-13, 9.75109e-13, 3.35666e-13, 3.57806e-13, 5.80017e-13, 3.56704e-13, 5.82922e-13, 3.87889e-13, 9.40234e-13, 6.51529e-13, 3.10739e-13, 7.28631e-13, 5.42934e-13, 5.10937e-13, 6.77585e-13, 2.89596e-13, 9.37336e-13, 1.12619e-13, 6.24809e-13, 1.9754e-14, 5.6047e-13, 4.71395e-13, 1.00163e-13, 9.26467e-13, 4.54221e-13, 7.46707e-13, 8.12512e-13, 7.19766e-13, 1.15493e-13, 5.16518e-13, 3.74628e-13, 6.00272e-13, 5.60742e-13, 4.83465e-13, 6.88265e-13, 8.42416e-13, 1.23533e-13, 2.46307e-13, 6.02184e-13, 1.19621e-13, 2.11899e-13, 2.369e-13, 8.18991e-13, 6.35244e-13, 2.24227e-13, 5.32174e-13, 2.63233e-13, 6.89934e-13, 6.95568e-14, 6.36287e-13, 1.39279e-14, 3.38425e-13, 9.44397e-13, 9.01372e-13, 8.40589e-13, 1.30085e-13, 3.99241e-13, 9.34649e-13, 8.31936e-13, 2.12785e-13, 8.27969e-13, 9.79731e-13, 8.50979e-13, 4.58418e-13, 4.68662e-13, 6.01553e-13, 1.38473e-13, 7.76258e-14, 4.43116e-13, 6.68033e-13, 7.0344e-14, 9.5118e-13, 3.23125e-13, 8.3377e-14, 7.5769e-13, 7.18118e-13, 4.0173e-13, 9.03811e-13, 6.97491e-13, 4.40708e-13, 5.83572e-13, 2.0214e-13, 3.63197e-13]
rok.File('results/demo-reactiveflow/k.pvd').write(k)


# exit()

# rho = rok.Constant(1000.0)  # water density (in units of kg/m3)
mu = rok.Constant(8.9e-4)  # water viscosity (in units of Pa*s)
# k = rok.Constant(1e-12)  # rock permeability (in units of m2)
f = rok.Constant(0.0)  # the source rate in the flow calculation


nsteps = 100       # the number of time steps
cfl = 0.3

D  = fire.Constant(1.0e-9)                # the diffusion coefficient (in units of m2/s)
v  = fire.Constant([1.0/week, 0.0])  # the fluid pore velocity (in units of m/s)
dt = 3*minute                           # the time step (in units of s)
# dt = 3*minute                           # the time step (in units of s)
# dt = 1200*minute                           # the time step (in units of s)
T  = 60.0 + 273.15                        # the temperature (in units of K)
P  = 100 * 1e5                            # the pressure (in units of Pa)


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
field.update()

# Initialize the Darcy flow solver
flow = rok.DarcyProblem(mesh)
flow.setFluidDensity(rho)
flow.setFluidViscosity(mu)
flow.setRockPermeability(k)
flow.setSourceRate(f)
flow.addPressureBC(1e5, 'right')
flow.addVelocityBC(v, 'left')
flow.addVelocityComponentBC(rok.Constant(0.0), 'y', 'bottom')
flow.addVelocityComponentBC(rok.Constant(0.0), 'y', 'top')


flow = rok.DarcySolver(flow)

# Initialize the chemical transport solver
transport = rok.ChemicalTransportSolver(field)
transport.addBoundaryCondition(state_bc, 1)  # 1 means left side in a rectangular mesh
transport.setVelocity([flow.u])
# transport.setVelocity([v])
transport.setDiffusion([D])

out_species = ['Ca++', 'Mg++', 'Calcite', 'Dolomite', 'CO2(aq)', 'HCO3-', 'Cl-', 'H2O(l)']
out_elements = ['H', 'O', 'C', 'Ca', 'Mg', 'Na', 'Cl']

nout = [fire.Function(V, name=name) for name in out_species]
bout = [fire.Function(V, name=name) for name in out_elements]

# Create the output file
file_species_amounts = fire.File('results/demo-reactiveflow/species-amounts.pvd')
file_element_amounts = fire.File('results/demo-reactiveflow/element-amounts.pvd')
file_porosity = fire.File('results/demo-reactiveflow/porosity.pvd')
file_volume = fire.File('results/demo-reactiveflow/volume.pvd')

t = 0.0
step = 0

# rho.assign(field.densities()[0])

flow.solve()

while step <= nsteps:
    print('Time: {:<5.2f} day ({}/{})'.format(t/day, step, nsteps))

    # For each selected species, output its molar amounts
    for f in nout:
        f.assign(field.speciesAmount(f.name()))

    # For each selected species, output its molar amounts
    for f in bout:
        f.assign(field.elementAmountInPhase(f.name(), 'Aqueous'))

    # file_species_amounts.write(*nout, flow.u, flow.p, rho)
    file_species_amounts.write(*nout, flow.u, flow.p)
    file_element_amounts.write(*bout)
    file_porosity.write(field.porosity())
    file_volume.write(field.volume())

    # Perform one transport step from `t` to `t + dt`
    field.setPressures(flow.p.dat.data)
    transport.step(field, dt)

    # rho.assign(field.densities()[0])

    # print(rho.dat.data)
    # exit()

    # Update the current time
    step += 1
    t += dt

