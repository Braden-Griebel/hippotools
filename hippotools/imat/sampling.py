# Core Module Imports
import warnings
from typing import Union

# External Imports
import cobra
from cobra.sampling import ACHRSampler, OptGPSampler
import pandas as pd
from dexom_python.default_parameter_values import DEFAULT_VALUES

# Local imports
from hippotools.imat.create_model import imat_model


def sample_imat(
    model: cobra.Model,
    reaction_weights: Union[pd.Series, dict],
    samples: int,
    thinning: int = None,
    processes: int = None,
    method: str = "optgp",
    **kwargs
):
    """
    Function to sample the solution space of an imat model
    :param model: Base model
    :type model: cobra.Model
    :param reaction_weights: Reaction weights for IMAT, 1 for highly expressed reactions, -1 for lowly expressed,
        0 otherwise
    :type reaction_weights: pd.Series | dict,
    :param samples: Number of samples to take
    :type samples: int
    :param method: Sampling method to use
    :type method: str
    :param thinning: Thinning factor for the sampling method
    :type thinning: int
    :param processes: Number of processes to use for OptGPSampler
    :type processes: int
    :param kwargs: Keyword arguments passed to imat_model
    :type kwargs: dict
    :return:
    """
    # THIS DOESN'T WORK, SAMPLING DOESN'T WORK WITH INTEGER MODELS
    # Create the imat model
    updated_model = imat_model(model=model, reaction_weights=reaction_weights, **kwargs)
    # Parse method
    method = _parse_method(method)
    # Create Sampler object
    if method == "optgp":
        sampler = OptGPSampler(
            model=updated_model, thinning=thinning, processes=processes
        )
    elif method == "achr":
        sampler = ACHRSampler(model=updated_model, thinning=thinning)
    else:
        raise ValueError("Could not parse method")
    # Sample
    samples_df = sampler.sample(samples)
    # Validate samples
    validity = sampler.validate(samples_df)
    # If any samples are invalid, warn and remove the invalid samples
    if not (validity == "v").all():
        warnings.warn("Some samples are invalid, removing them")
        samples_df = samples_df[validity == "v"]
    # Return the samples
    return samples_df


def _parse_method(method):
    if method.lower() in [
        "a",
        "achr",
        "artificial centering hit-and-run",
        "artificial centering hit-and-run sampler",
    ]:
        return "achr"
    elif method.lower() in ["o", "opt", "optg", "optgp"]:
        return "optgp"
    else:
        raise ValueError("Could not parse method")
