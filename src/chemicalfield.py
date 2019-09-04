from reaktoro import *
from firedrake import *
import numpy as np

class _ChemicalField(object):

    def __init__(self, system, function_space):
        # Initialize the ChemicalSystem instance
        self.system = system

        # Initialize the partition instance
        self.partition = Partition(system)

        # Initialize the function space where the chemical field is defined
        self.function_space = function_space

        # Initialize the mesh member
        self.mesh = function_space.mesh()

        # Initialize the number of degree-of-freedoms
        self.num_dofs = len(self.function_space.dof_count)

        # Initialize the chemical state of every degree-of-freedom
        self.states = [ChemicalState(system) for i in range(self.num_dofs)]

        # Initialize the indices of fluid and solid phases
        self.iphases_fluid = partition.indicesFluidPhases()
        self.iphases_solid = partition.indicesSolidPhases()

        # Initialize the Function instances for the saturation field of each fluid phase
        self.sat = [Function(function_space) for i in self.iphases_fluid]

        # Set the names of the saturation Function instances
        for (func, iphase) in zip(self.sat, self.iphases_fluid):
            name = self.system.phase(iphase).name()
            name = 'Saturation[%s]' % name
            func.rename(name, name)

        # Initialize the auxiliary array used for setting self.phi
        self.sat_values = np.zeros((len(self.iphases_fluid), self.num_dofs))

        # Initialize the Function instance for the phi field
        self.phi = Function(function_space)

        # Set the name of the phi Function instance
        self.phi.rename('Porosity', 'Porosity')

        # Initialize the auxiliary array used for setting self.phi
        self.phi_values = np.zeros(self.num_dofs)

        # Initialize the auxliary Function instance for output purposes
        self.out = Function(function_space)

        # Initialize the auxiliary array for setting Function instances
        self.values = np.zeros(self.num_dofs)


    def fill(self, state):
        for k in range(self.num_dofs):
            self.states[k].assign(state)


    def setTemperatures(self, temperatures):
        for k in range(self.num_dofs):
            self.states[k].setTemperature(temperatures[k])


    def setPressures(self, pressures):
        for k in range(self.num_dofs):
            self.states[k].setPressure(pressures[k])


    def update(self):
        for k in range(self.num_dofs):
            properties = self.states[k].properties()
            v = properties.phaseVolumes().val
            volume_fluid = sum([v[i] for i in self.iphases_fluid])
            volume_solid = sum([v[i] for i in self.iphases_solid])
            self.phi_values[k] = 1.0 - volume_solid
            self.sat_values[:, k] = [v[i]/volume_fluid for i in self.iphases_fluid]
        self.phi.vector()[:] = self.phi_values
        for i in range(len(self.iphases_fluid)):
            self.sat[i].vector()[:] = self.sat_values[i]


    def elementAmounts(self, b):
        for i in range(self.num_dofs):
            vec = self.states[i].elementAmounts()
            b[:, i] = vec
        return b


    def elementAmountsInSpecies(self, indices, b):
        for i in range(self.num_dofs):
            vec = self.states[i].elementAmountsInSpecies(indices)
            b[:, i] = vec
        return b


    def speciesAmount(self, species):
        ispecies = self.system.indexSpecies(species)
        for k in range(self.num_dofs):
            self.values[k] = self.states[k].speciesAmount(ispecies)
        self.out.vector()[:] = self.values
        self.out.rename(species, species)
        return self.out


    def elementAmountInPhase(self, element, phase):
        ielement = self.system.indexElement(element)
        iphase = self.system.indexPhase(phase)
        for k in range(self.num_dofs):
            self.values[k] = self.states[k].elementAmountInPhase(ielement, iphase)
        self.out.vector()[:] = self.values
        self.out.rename(element, element)
        return self.out


    def volume(self):
        for k in range(self.num_dofs):
            properties = self.states[k].properties()
            v = properties.phaseVolumes().val
            self.values[k] = sum(v)
        self.out.vector()[:] = self.values
        self.out.rename('Volume', 'Volume')
        return self.out


#     def ph(self, species):
#         ispecies = self.system.indexSpecies(species)
#         for k in range(self.num_dofs):
#             self.values[k] = self.states[k].speciesAmount(ispecies)
#         self.out.vector()[:] = self.values
#         return self.out



class ChemicalField(object):

    def __init__(self, system, partition, function_space):
        self.pimpl = _ChemicalField(system, partition, function_space)


    def fill(self, state):
        self.pimpl.fill(state)


    def setTemperatures(self, temperatures):
        self.pimpl.setTemperatures(temperatures)


    def setPressures(self, pressures):
        self.pimpl.setPressures(pressures)


    def update(self):
        self.pimpl.update()


    def elementAmounts(self, b):
        self.pimpl.elementAmounts(b)


    def elementAmountsInSpecies(self, indices, b):
        self.pimpl.elementAmountsInSpecies(indices, b)


    def states(self):
        return self.pimpl.states


    def system(self):
        return self.pimpl.system


    def partition(self):
        return self.pimpl.partition


    def functionSpace(self):
        return self.pimpl.function_space


    def porosity(self):
        return self.pimpl.phi


    def volume(self):
        return self.pimpl.volume()


    def saturations(self):
        return self.pimpl.sat


    def speciesAmount(self, species):
        return self.pimpl.speciesAmount(species)


    def elementAmountInPhase(self, element, phase):
        return self.pimpl.elementAmountInPhase(element, phase)

