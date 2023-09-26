# Core module import
import pathlib
import os
import unittest

# External imports
import cobra
from dexom_python.model_functions import load_reaction_weights
from dexom_python.imat_functions import imat
from dexom_python.default_parameter_values import DEFAULT_VALUES
import numpy as np
import pandas as pd

# Local imports
from hippotools.imat.utils import calc_imat_obj, validate_imat_flux
from hippotools.utils import read_model


class TestImatFluxFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data_dir = str(pathlib.Path(__file__).parent.parent.joinpath("data"))
        cls.model = read_model(os.path.join(cls.data_dir, "test_model.json"))
        cls.reaction_weights = load_reaction_weights(
            os.path.join(cls.data_dir, "test_model_weights.csv")
        )
        imat_solution = imat(
            model=cls.model,
            reaction_weights=cls.reaction_weights,
            epsilon=DEFAULT_VALUES["epsilon"],
            threshold=DEFAULT_VALUES["threshold"],
        )
        cls.fluxes = imat_solution.fluxes
        cls.imat_obj = imat_solution.objective_value

    def test_calc_imat_obj(self):
        imat_obj_calc = calc_imat_obj(
            flux=self.fluxes,
            reaction_weights=self.reaction_weights,
            imat_obj=self.imat_obj,
            epsilon=DEFAULT_VALUES["epsilon"],
        )
        self.assertTrue(np.isclose(imat_obj_calc, self.imat_obj))

    def test_validate_imat_obj(self):
        self.assertTrue(
            validate_imat_flux(
                flux=self.fluxes,
                reaction_weights=self.reaction_weights,
                imat_obj=self.imat_obj,
                epsilon=DEFAULT_VALUES["epsilon"],
                obj_tol=DEFAULT_VALUES["obj_tol"],
            )
        )
        self.assertFalse(
            validate_imat_flux(
                flux=self.fluxes,
                reaction_weights=self.reaction_weights,
                imat_obj=self.imat_obj,
                epsilon=0,
                obj_tol=DEFAULT_VALUES["obj_tol"],
            )
        )
        self.assertFalse(
            validate_imat_flux(
                flux=self.fluxes / 100000,
                reaction_weights=self.reaction_weights,
                imat_obj=self.imat_obj,
                epsilon=DEFAULT_VALUES["epsilon"],
                obj_tol=DEFAULT_VALUES["obj_tol"],
            )
        )


if __name__ == "__main__":
    unittest.main()
