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


def rough_porosity(porosity_field, low_cut, high_cut, value_at_low_cut, value_at_high_cut):
    new_porosity = fire.Function(porosity_field.function_space()).project(
        _rough_porosity_function(
            porosity_value=porosity_field,
            low_cut=low_cut,
            high_cut=high_cut,
            value_at_low_cut=value_at_low_cut,
            value_at_high_cut=value_at_high_cut,
        )
    )
    return new_porosity


def _rough_porosity_function(
    porosity_value, low_cut, high_cut, value_at_low_cut, value_at_high_cut
):
    if porosity_value < low_cut:
        return value_at_low_cut
    elif porosity_value < high_cut:
        mid_range_porosity = (porosity_value - low_cut) / (high_cut - low_cut)
        return mid_range_porosity
    elif high_cut <= porosity_value <= 1:
        return value_at_high_cut
    else:
        return 1
