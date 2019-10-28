import firedrake as fire
import numpy as np
import reaktoro as rkt
import rok


def test_chemicaltransport_step(num_regression):

    nx, ny, nz = 5, 5, 5  # number of cells along x, y, and z
    nsteps = 10  # number of time steps

    velocity = fire.Constant([1.0e-2, 0.0, 0.0])  # the velocity (in units of m/s)
    diffusion = fire.Constant(1.0e-4)  # the diffusion coefficient (in units of m2/s)
    dt = 1.0  # the time step (in units of s)
    T = 60.0 + 273.15  # the temperature (in units of K)
    P = 100 * 1e5  # the pressure (in units of Pa)

    mesh = fire.UnitCubeMesh(nx, ny, nz)
    V = fire.FunctionSpace(mesh, "CG", 1)

    # Initialise the database
    database = rkt.Database("supcrt98.xml")

    # Initialise the chemical editor
    editor = rkt.ChemicalEditor(database)
    editor.addAqueousPhase("H2O(l) H+ OH- Na+ Cl- Ca++ Mg++ HCO3- CO2(aq) CO3--")
    editor.addMineralPhase("Quartz")
    editor.addMineralPhase("Calcite")
    editor.addMineralPhase("Dolomite")

    # Initialise the chemical system
    system = rkt.ChemicalSystem(editor)

    # Define the initial condition of the reactive transport modeling problem
    problem_ic = rkt.EquilibriumProblem(system)
    problem_ic.setTemperature(T)
    problem_ic.setPressure(P)
    problem_ic.add("H2O", 1.0, "kg")
    problem_ic.add("NaCl", 0.7, "mol")
    problem_ic.add("CaCO3", 10, "mol")
    problem_ic.add("SiO2", 10, "mol")

    # Define the boundary condition of the reactive transport modeling problem
    problem_bc = rkt.EquilibriumProblem(system)
    problem_bc.setTemperature(T)
    problem_bc.setPressure(P)
    problem_bc.add("H2O", 1.0, "kg")
    problem_bc.add("NaCl", 0.90, "mol")
    problem_bc.add("MgCl2", 0.05, "mol")
    problem_bc.add("CaCl2", 0.01, "mol")
    problem_bc.add("CO2", 0.75, "mol")

    # Calculate the equilibrium states for the initial and boundary conditions
    state_ic = rkt.equilibrate(problem_ic)
    state_bc = rkt.equilibrate(problem_bc)

    # Scale the volumes of the phases in the initial condition such that their sum is 1 m3
    state_ic.scalePhaseVolume("Aqueous", 0.1, "m3")
    state_ic.scalePhaseVolume("Quartz", 0.882, "m3")
    state_ic.scalePhaseVolume("Calcite", 0.018, "m3")

    # Scale the volume of the boundary equilibrium state to 1 m3
    state_bc.scaleVolume(1.0)

    # Initialise the chemical field
    field = rok.ChemicalField(system, V)
    field.fill(state_ic)

    # Initialize the chemical transport solver
    transport = rok.ChemicalTransportSolver(field)
    transport.addBoundaryCondition(state_bc, 1)  # 1 means left side
    transport.setVelocity([velocity])
    transport.setDiffusion([diffusion])

    species_names_for_output = [
        "Ca++",
        "Mg++",
        "Calcite",
        "Dolomite",
        "CO2(aq)",
        "HCO3-",
        "Cl-",
        "H2O(l)",
    ]
    element_names_for_output = ["H", "O", "C", "Ca", "Mg", "Na", "Cl"]

    species_amount_functions = [fire.Function(V, name=name) for name in species_names_for_output]
    element_amount_functions = [fire.Function(V, name=name) for name in element_names_for_output]

    step = 0

    species_data = []
    element_data = []

    while step <= nsteps:
        transport.step(field, dt)

        # For each selected species, output its molar amounts
        for f in species_amount_functions:
            f.assign(field.speciesAmount(f.name()))
            species_data.append(f.dat.data)

        # For each selected species, output its molar amounts
        for f in element_amount_functions:
            f.assign(field.elementAmountInPhase(f.name(), "Aqueous"))
            element_data.append(f.dat.data)

        step += 1

    species_data = {
        "n({})(step={})".format(name, i): u
        for i, (name, u) in enumerate(zip(species_names_for_output, species_data))
    }
    element_data = {
        "b({})(step={})".format(name, i): u
        for i, (name, u) in enumerate(zip(element_names_for_output, element_data))
    }
    data = {**species_data, **element_data}

    num_regression.check(data)
