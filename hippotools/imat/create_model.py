# Core module import
from typing import Union

# External Imports
import cobra
import pandas as pd
from dexom_python.default_parameter_values import DEFAULT_VALUES
from dexom_python.enum_functions.enumeration import create_enum_variables
from dexom_python.enum_functions.maxdist_functions import create_maxdist_constraint
from dexom_python.imat_functions import (
    imat,
)
from dexom_python.model_functions import check_threshold_tolerance

# Local imports
from hippotools.utils import HiddenPrints


def imat_model(
    model: cobra.Model,
    reaction_weights: Union[pd.Series, dict],
    prev_solution: cobra.Solution = None,
    epsilon=DEFAULT_VALUES["epsilon"],
    threshold=DEFAULT_VALUES["threshold"],
    obj_tol=DEFAULT_VALUES["obj_tol"],
):
    """
    Function to add imat constraints to a model, ensuring that all solutions to the model are consistent with the
    provided reaction weights. This function will modify the model in place, and will return the modified model.
    :param model: A cobrapy model to create the imat model from
    :type model: cobra.Model
    :param reaction_weights: Reactions weights, keys are reaction IDs, values are reaction weights
    :type reaction_weights: pd.Series | dict
    :param prev_solution: Previous imat solution (if none, will generate a new one)
    :type prev_solution: pd.Series
    :param epsilon: Activation threshold for highly expressed reactions
    :type epsilon: float
    :param threshold: Activation threshold for all reactions
    :type threshold: float
    :param obj_tol: Tolerance for the imat objective constraint
    :type obj_tol: float
    :return: Model with added imat constraints
    :rtype: cobra.Model
    """
    # Make sure the reaction weights are a dict
    if reaction_weights is pd.Series:
        reaction_weights = reaction_weights.to_dict()
    # Check that the threshold is within the tolerance of the solution
    check_threshold_tolerance(model=model, epsilon=epsilon, threshold=threshold)
    if reaction_weights is pd.Series:
        reaction_weights = reaction_weights.to_dict()
    # Run imat to get the solution if none provided
    if prev_solution is None:
        with HiddenPrints():
            prev_solution = imat(
                model,
                reaction_weights,
                epsilon=epsilon,
                threshold=threshold,
                full=False,
            )
    else:
        model = create_enum_variables(
            model=model,
            reaction_weights=reaction_weights,
            eps=epsilon,
            thr=threshold,
            full=False,
        )
    opt_const = create_maxdist_constraint(
        model=model,
        reaction_weights=reaction_weights,
        prev_sol=prev_solution,
        obj_tol=obj_tol,
        name="imat_opt_const",
        full=False,
    )
    model.solver.add(opt_const)
    return model
