# Core module imports
import os
import pathlib
import unittest

import pandas as pd

# External Imports
from dexom_python.model_functions import load_reaction_weights
from dexom_python.imat_functions import imat

# Local imports
from hippotools.utils import read_model
from hippotools.imat.sampling import sample_imat, _parse_method
from hippotools.imat.utils import validate_imat_flux


class TestSampling(unittest.TestCase):
    def test_parse_method(self):
        self.assertEqual(_parse_method("optgp"), "optgp")
        self.assertEqual(_parse_method("OptGP"), "optgp")
        self.assertEqual(_parse_method("o"), "optgp")
        self.assertEqual(_parse_method("achr"), "achr")
        self.assertEqual(_parse_method("Achr"), "achr")
        self.assertEqual(_parse_method("a"), "achr")

    def test_sampling(self):
        data_path = str(pathlib.Path(__file__).parent.parent.joinpath("data"))
        test_model = read_model(os.path.join(data_path, "test_model.json"))
        copy_model = read_model(os.path.join(data_path, "test_model.json"))
        rxn_weights = load_reaction_weights(
            os.path.join(data_path, "test_model_weights.csv")
        )
        samples = sample_imat(
            model=test_model,
            reaction_weights=rxn_weights,
            samples=100,
            thinning=10,
            method="optgp",
        )
        imat_obj = imat(test_model, rxn_weights).objective_value
        self.assertTrue(samples is pd.DataFrame)
        valid_flux = samples.apply(
            lambda x: validate_imat_flux(
                x, reaction_weights=rxn_weights, imat_obj=imat_obj
            ),
            axis=1,
        )
        self.assertTrue(valid_flux.all())


if __name__ == "__main__":
    unittest.main()
