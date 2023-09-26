# Core module imports
from typing import Union

# External Imports
import cobra
from dexom_python.default_parameter_values import DEFAULT_VALUES
import numpy as np
import pandas as pd
import six


def validate_imat_flux(
    flux: pd.Series,
    reaction_weights: Union[pd.Series, dict],
    imat_obj: float,
    epsilon: float = DEFAULT_VALUES["epsilon"],
    obj_tol: float = DEFAULT_VALUES["obj_tol"],
):
    """
    Function to validate that a flux solution is consistent with the IMAT objective
    :param flux: Flux values
    :type flux: pd.Series
    :param reaction_weights: Reaction weights for IMAT, 1 for highly expressed reactions, -1 for lowly expressed,
        0 otherwise
    :type reaction_weights: pd.Series | dict
    :param imat_obj: IMAT objective value
    :type imat_obj: float
    :param obj_tol: Tolerance for the imat objective constraint, a proportion of the IMAT objective value
        validation will pass if the calculated objective is at least (imat_obj * (1 - obj_tol))
    :type obj_tol: float
    :param epsilon: Activation threshold for highly expressed reactions
    :return: True if the flux solution is consistent with the IMAT objective, False otherwise
    """
    imat_obj_calc = calc_imat_obj(
        flux=flux, reaction_weights=reaction_weights, imat_obj=imat_obj, epsilon=epsilon
    )
    if imat_obj_calc < imat_obj * (1 - obj_tol):
        return False
    else:
        return True


def calc_imat_obj(
    flux: pd.Series,
    reaction_weights: Union[pd.Series, dict],
    imat_obj: float,
    epsilon: float = DEFAULT_VALUES["epsilon"],
):
    """
    Function to calculate the IMAT objective value from flux values
    :param flux: Reaction flux values
    :param reaction_weights: Weights for IMAT, 1 for highly expressed reactions, -1 for lowly expressed, 0 otherwise
    :param imat_obj:
    :param epsilon:
    :return:
    """
    high_active = 0
    low_inactive = 0
    # This can be significantly optimized, but correctness is more important than speed for now
    for rid, weight in six.iteritems(reaction_weights):
        # If reaction is highly expressed
        if weight > 0:
            if flux[rid] > epsilon or flux[rid] < -epsilon:
                high_active += 1
        elif weight < 0:
            if np.abs(flux[rid]) < epsilon:
                low_inactive += 1
    return high_active + low_inactive
