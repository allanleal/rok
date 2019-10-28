import firedrake as fire
from gstools import SRF, Gaussian, TPLStable, Exponential
from scipy.interpolate import RegularGridInterpolator
import numpy as np


def rescale_field(x_input, x_new_min, x_new_max):
    x_min_input = x_input.min()
    x_max_input = x_input.max()
    return (x_input - x_min_input) / (x_max_input - x_min_input) * (
        x_new_max - x_new_min
    ) + x_new_min


def random_field_generator(
    function_space, minval=0, maxval=1, var=1e-2, len_scale=1, len_low=0, seed=20170519
):
    mesh = function_space.mesh()
    dim = mesh.geometric_dimension()

    model = TPLStable(dim=dim, var=var, len_scale=len_scale, len_low=len_low)
    # model = Exponential(dim=dim, var=var, len_scale=len_scale)  # FIXME: Diego, the exponential model, instead of TPLStable, resulted in some chemical equilibrium errors (returning to previous at the moment)

    srf = SRF(model, seed=seed)

    ndofs = function_space.dim()
    size = int(ndofs ** (1 / dim))

    xyz = [range(size) for _ in range(dim)]

    data = srf(xyz, mesh_type="structured")

    # The min and max values of x,y,z in the mesh
    xyz_min = [mesh.coordinates.dat.data[:, i].min() for i in range(dim)]
    xyz_max = [mesh.coordinates.dat.data[:, i].max() for i in range(dim)]

    xyz = [np.linspace(xyz_min[i], xyz_max[i], size) for i in range(dim)]

    datainterp = RegularGridInterpolator(xyz, data)

    # Now make the VectorFunctionSpace corresponding to V.
    W = fire.VectorFunctionSpace(mesh, function_space.ufl_element())

    # Next, interpolate the coordinates onto the nodes of W.
    X = fire.interpolate(mesh.coordinates, W)

    # Make an output function.
    k = fire.Function(function_space, name="Permeability")

    # Use the external data function to interpolate the values of f.
    kdata = np.array([datainterp(xyz) for xyz in X.dat.data_ro])

    # k.dat.data[:] = mydata(X.dat.data_ro)
    k.dat.data[:] = rescale_field(kdata, minval, maxval)[:, 0]

    return k
