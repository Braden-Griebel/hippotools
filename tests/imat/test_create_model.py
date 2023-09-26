# Standard Library Imports
import os
import pathlib
import unittest

# External Imports
from dexom_python.default_parameter_values import DEFAULT_VALUES
from dexom_python.imat_functions import imat
import numpy as np
import pandas as pd
import six

# Local Imports
from hippotools.utils import read_model
from hippotools.imat.create_model import imat_model
from hippotools.utils import HiddenPrints


class TestCreateModel(unittest.TestCase):
    def test_create_model(self):
        data_path = str(pathlib.Path(__file__).parent.parent.joinpath("data"))
        test_model = read_model(os.path.join(data_path, "test_model.json"))
        rxn_weights = pd.read_csv(
            os.path.join(data_path, "test_model_weights.csv"),
            usecols=[1, 2],
            index_col=0,
        )
        test_weights = rxn_weights["weights"].to_dict()
        with HiddenPrints():
            imat_solution = imat(model=test_model, reaction_weights=test_weights)
        imat_obj = imat_solution.objective_value
        imat_test_model = imat_model(
            model=test_model, reaction_weights=test_weights, prev_solution=imat_solution
        )
        imat_test_model.objective = "R_a_f"
        test_solution = imat_test_model.optimize()
        eps = DEFAULT_VALUES["epsilon"]
        thr = DEFAULT_VALUES["threshold"]
        high_active = 0
        low_inactive = 0
        for rid, weight in six.iteritems(test_weights):
            # If reaction is highly expressed
            if weight > 0:
                if test_solution.fluxes[rid] > eps or test_solution.fluxes[rid] < -eps:
                    high_active += 1
            elif weight < 0:
                if np.abs(test_solution.fluxes[rid]) < eps:
                    low_inactive += 1
        self.assertTrue(np.isclose(high_active + low_inactive, imat_obj))
        # Now test without a previous solution
        # Reread model
        test_model = read_model(os.path.join(data_path, "test_model.json"))
        imat_test_model = imat_model(
            model=test_model, reaction_weights=test_weights, prev_solution=None
        )
        imat_test_model.objective = "R_a_f"
        test_solution = imat_test_model.optimize()
        eps = DEFAULT_VALUES["epsilon"]
        thr = DEFAULT_VALUES["threshold"]
        high_active = 0
        low_inactive = 0
        for rid, weight in six.iteritems(test_weights):
            # If reaction is highly expressed
            if weight > 0:
                if test_solution.fluxes[rid] > eps or test_solution.fluxes[rid] < -eps:
                    high_active += 1
            elif weight < 0:
                if np.abs(test_solution.fluxes[rid]) < eps:
                    low_inactive += 1
        self.assertTrue(np.isclose(high_active + low_inactive, imat_obj))


if __name__ == "__main__":
    unittest.main()
