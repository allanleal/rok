import numpy as np
import rok
import matplotlib.pyplot as plt

mesh = rok.RectangleMesh(100, 100, 1, 1)
rok.plot(mesh)
plt.show()

phi_space = rok.FunctionSpace(mesh, "DG", 0)

porosity_min_value = 0.05
porosity_max_value = 0.6

phi = rok.porosity(phi_space, minval=porosity_min_value, maxval=porosity_max_value, len_scale=15)
rok.plot(phi)
plt.show()

porosity_low_cut = 0.2
porosity_high_cut = 0.4
# Values smaller than porosity_low_cut will be converted to porosity_low_cut. Likewise, values greater
# porosity_high_cut will be truncated to porosity_high_cut.
rough_phi_1 = rok.rough_porosity(phi, porosity_low_cut, porosity_high_cut)
rok.plot(rough_phi_1)
plt.show()

# Alternatively, we can set the value for the truncations. Note that this field has stronger discontinuities
# when compared with the former.
rough_phi_2 = rok.rough_porosity(
    phi,
    porosity_low_cut,
    porosity_high_cut,
    value_at_low_cut=porosity_min_value,
    value_at_high_cut=porosity_max_value,
)
rok.plot(rough_phi_2)
plt.show()
