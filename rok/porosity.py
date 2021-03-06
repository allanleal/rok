import firedrake as fire
from rok import random_field
from gstools import SRF, Gaussian, TPLStable, Exponential
from scipy.interpolate import RegularGridInterpolator
import numpy as np


def porosity(
    function_space, minval=0.0, maxval=1.0, var=1e-2, len_scale=1, len_low=0, seed=20170519
):
    return random_field.random_field_generator(
        function_space,
        minval=minval,
        maxval=maxval,
        var=var,
        len_scale=len_scale,
        len_low=len_low,
        seed=seed,
    )


def rough_porosity(
    porosity_field, low_cut, high_cut, value_at_low_cut=None, value_at_high_cut=None
):
    if value_at_low_cut is None:
        value_at_low_cut = low_cut
    if value_at_high_cut is None:
        value_at_high_cut = high_cut

    rough_porosity_field = _rough_porosity_transformation(
        porosity_field.dat.data_ro[:], low_cut, high_cut, value_at_low_cut, value_at_high_cut
    )
    new_porosity = fire.Function(porosity_field.function_space(), val=rough_porosity_field)
    return new_porosity


def _rough_porosity_transformation(
    porosity_value, low_cut, high_cut, value_at_low_cut, value_at_high_cut
):
    new_porosity = np.where(
        porosity_value < low_cut,
        value_at_low_cut,
        np.where(
            porosity_value < high_cut,
            porosity_value,
            np.where(porosity_value < 1, value_at_high_cut, value_at_low_cut),
        ),
    )
    return new_porosity
