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


def rough_porosity(porosity_space, low_cut, high_cut, value_at_low_cut, value_at_high_cut):
    NotImplementedError("To be implemented.")
