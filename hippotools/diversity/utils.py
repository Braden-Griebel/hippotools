"""
Utility functions for diversity methods
"""

# Standard library imports
import re

# Local imports
from hippotools.diversity.iterator import (
    DiversityEnumIterator,
    MaxDistEnumIterator,
    IcutEnumIterator,
)


# region parsing methods
def parse_enum_method(method: str) -> str:
    """
    The parse_enum_method function takes a string and returns the corresponding enumeration method.

    :param method: Method string to parse
    :type method: str
    :return: A string representing the enumeration method
    :rtype: str
    """
    if method[0:3].lower() == "div":
        return "diversity"
    elif method[0:4].lower() == "icut":
        return "icut"
    elif method[0:3].lower() == "max":
        return "max-dist"
    else:
        raise ValueError("Couldn't Parse Enumeration Method: %s" % method)


def parse_model_method(method: str) -> str:
    """
    The parse_model_method function takes a string and parses it into one of the following strings:
        - enforce_active
        - enforce_inactive
        - enforce_inactive_off
        - enforce_off

    :param method: Specify the method used to train the model
    :type method: str
    :return: A string describing the model method
    :rtype: str
    :doc-author: Trelent
    """
    inactive_pattern = re.compile("inact[ive]*", re.IGNORECASE)
    active_pattern = re.compile("(?<!in)act[ive]*", re.IGNORECASE)
    off_pattern = re.compile("off", re.IGNORECASE)
    enforce_pattern = re.compile("enf[orce]*", re.IGNORECASE)
    both_pattern = re.compile("both", re.IGNORECASE)
    active_flag = bool(active_pattern.search(method))
    inactive_flag = bool(inactive_pattern.search(method))
    off_flag = bool(off_pattern.search(method))
    enforce_flag = bool(enforce_pattern.search(method))
    both_flag = bool(both_pattern.search(method))
    if enforce_flag:
        if both_flag:
            if not off_flag:
                return "enforce_both"
            else:
                raise NotImplementedError(
                    "Enforcing both and off constraints not currently implemented"
                )
        elif active_flag:
            if not (inactive_flag or off_flag):
                return "enforce_active"
            else:
                if inactive_flag:
                    raise ValueError("Can't enforce active and inactive simultaneously")
                if off_flag:
                    raise NotImplementedError(
                        "Enforcing active and off simultaneously not currently implemented"
                    )
        elif inactive_flag:
            if off_flag:
                return "enforce_inactive_off"
            else:
                return "enforce_inactive"
        elif off_flag:
            return "enforce_off"
        else:
            raise ValueError(
                "Couldn't parse context specific model method: %s" % method
            )
    else:
        raise ValueError("Couldn't parse context specific model method: %s" % method)


# endregion
def create_iterator(model, reaction_weights, enum_method, **kwargs):
    """
    Method to create an iterator from a model and reactions weights, using enum_method

    :param model: Base metabolic model to use for creating context specific models'
    :type model: cobra.Model
    :param reaction_weights: Reaction weights for the enumeration method, indexed by reaction id, where values of -1
        mean the reaction has a low expression level, values of 1 mean the reaction has a high expression level,
        and 0 is for all other reactions
    :type reaction_weights: pd.Series
    :param enum_method: Specify the enumeration method to use (i.e. diversity, maxdist, or icut)
    :type enum_method: str
    :param kwargs: Keyword dictionary passed to enum_method iterator class init function, see documentation
        for the desired class for information on possible arguments
    :type kwargs: dict
    :return: Class for iterating through the enumerated context specific solutions
    :rtype: DiversityEnumIterator, MaxDistEnumIterator, or IcutEnumIterator
    """
    enum_method = parse_enum_method(enum_method)
    if enum_method == "diversity":
        iterator = DiversityEnumIterator(
            model=model, reaction_weights=reaction_weights, **kwargs
        )
    elif enum_method == "maxdist":
        iterator = MaxDistEnumIterator(
            model=model, reaction_weights=reaction_weights, **kwargs
        )
    elif enum_method == "icut":
        iterator = IcutEnumIterator(
            model=model, reaction_weights=reaction_weights, **kwargs
        )
    else:
        raise ValueError("Couldn't parse Enumeration Method")
    return iterator
