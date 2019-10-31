import firedrake as fire
from rok import random_field
from gstools import SRF, Gaussian, TPLStable, Exponential
from scipy.interpolate import RegularGridInterpolator
import numpy as np


def permeability(
    function_space, minval=1e-14, maxval=1e-10, var=1e-2, len_scale=10, len_low=0, seed=20170519
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
