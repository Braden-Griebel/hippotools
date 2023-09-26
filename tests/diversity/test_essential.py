# Standard Library Imports
import unittest

# External Imports
import pandas as pd

# Local Imports
from hippotools.diversity.essential import (
    aggstrat_all,
    aggstrat_majority,
    aggstrat_any,
    compute_essentiality,
)


class TestAggStratFunctions(unittest.TestCase):
    """
    Unittest test case for the Aggregation strategy Functions
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up essentiality series for testing
        """
        cls.all_true = pd.Series([True, True, True, True], dtype="boolean")
        cls.all_true_na = pd.Series([True, True, True, True, pd.NA], dtype="boolean")
        cls.all_false = pd.Series([False, False, False, False], dtype="boolean")
        cls.all_false_na = pd.Series(
            [False, False, False, False, pd.NA], dtype="boolean"
        )
        cls.most_true = pd.Series([True, True, True, False], dtype="boolean")
        cls.most_true_na = pd.Series([True, True, True, False, pd.NA], dtype="boolean")
        cls.most_false = pd.Series([False, False, False, False, True], dtype="boolean")
        cls.most_false_na = pd.Series(
            [False, False, False, False, True, pd.NA], dtype="boolean"
        )
        cls.half_true = pd.Series(
            [True, True, True, False, False, False], dtype="boolean"
        )
        cls.half_true_na = pd.Series(
            [True, True, True, False, False, False, pd.NA], dtype="boolean"
        )

    def test_strat_all(self):
        """
        Test the aggstrat_all function
        """
        # If all true should be true
        self.assertTrue(aggstrat_all(self.all_true))
        # If all are true except NAs, should return NA
        self.assertTrue(pd.isna(aggstrat_all(self.all_true_na)))
        # If all are false should return false
        self.assertFalse(aggstrat_all(self.all_false))
        # If any are false should return false
        self.assertFalse(aggstrat_all(self.most_true))
        # Even if there is an NA
        self.assertFalse(aggstrat_all(self.most_true_na))
        # If all are true except NAs, and ignore_na is true, should return true
        self.assertTrue(aggstrat_all(self.all_true_na, ignore_na=True))
        # If there is a single false, and ignore_na is true, should return False
        self.assertFalse(aggstrat_all(self.most_true_na, ignore_na=True))

    def test_strat_any(self):
        """
        Test the aggstrat_any function
        """
        # If any Trues are present, should return True
        self.assertTrue(aggstrat_any(self.most_true))
        self.assertTrue(aggstrat_any(self.most_true_na))
        self.assertTrue(aggstrat_any(self.most_false))
        self.assertTrue(aggstrat_any(self.most_false_na))
        # If all False, should return False
        self.assertFalse(aggstrat_any(self.all_false))
        # Should return NA if all False with an NA
        self.assertTrue(pd.isna(aggstrat_any(self.all_false_na)))
        # Unless ignore na is True
        self.assertFalse(aggstrat_any(self.all_false_na, ignore_na=True))

    def test_strat_majority(self):
        """
        Test the aggstrat_majority function
        """
        # If all or majority are true, should return True
        self.assertTrue(aggstrat_majority(self.all_true))
        self.assertTrue(aggstrat_majority(self.all_true_na))
        self.assertTrue(aggstrat_majority(self.most_true))
        self.assertTrue(aggstrat_majority(self.most_true_na))
        # If all or majority are false, should return false
        self.assertFalse(aggstrat_majority(self.all_false))
        self.assertFalse(aggstrat_majority(self.all_false_na))
        self.assertFalse(aggstrat_majority(self.most_false))
        self.assertFalse(aggstrat_majority(self.most_false_na))
        # If half false, should return false
        self.assertFalse(aggstrat_majority(self.half_true))
        # Unless NA is present
        self.assertTrue(pd.isna(aggstrat_majority(self.half_true_na)))
        # If half false and half true other than NA, should return false
        self.assertFalse(aggstrat_majority(self.half_true_na, ignore_na=True))


class TestComputeEssentiality(unittest.TestCase):
    """
    Unittest test case for the compute_essentiality function
    """

    @classmethod
    def setUpClass(cls):
        """
        Create dataframe for testing
        """
        cls.test_df = pd.DataFrame(
            {
                "iter_1": [True, False, True, True, True, False, True],
                "iter_2": [True, False, True, False, True, False, True],
                "iter_3": [True, False, True, False, False, False, True],
                "iter_4": [True, False, True, True, False, False, True],
                "iter_5": [True, False, False, True, False, False, True],
                "iter_6": [True, False, False, pd.NA, pd.NA, pd.NA, True],
                "iter_7": [True, False, False, True, pd.NA, False, pd.NA],
                "iter_8": [True, False, False, pd.NA, pd.NA, False, True],
            },
            index=[
                "gene_1",
                "gene_2",
                "gene_3",
                "gene_4",
                "gene_5",
                "gene_6",
                "gene_7",
            ],
            dtype="boolean",
        )

    def test_compute_essentiality(self):
        any_no_ignore = compute_essentiality(
            consensus_essentiality_df=self.test_df, aggregation_func=aggstrat_any
        )
        any_ignore_na = compute_essentiality(
            consensus_essentiality_df=self.test_df,
            aggregation_func=aggstrat_any,
            ignore_na=True,
        )
        all_no_ignore = compute_essentiality(
            consensus_essentiality_df=self.test_df, aggregation_func=aggstrat_all
        )
        all_ignore_na = compute_essentiality(
            consensus_essentiality_df=self.test_df,
            aggregation_func=aggstrat_all,
            ignore_na=True,
        )
        maj_no_ignore = compute_essentiality(
            consensus_essentiality_df=self.test_df,
            aggregation_func=aggstrat_majority,
        )
        maj_ignore_na = compute_essentiality(
            consensus_essentiality_df=self.test_df,
            aggregation_func=aggstrat_majority,
            ignore_na=True,
        )
        expected_index = pd.Index(
            ["gene_1", "gene_2", "gene_3", "gene_4", "gene_5", "gene_6", "gene_7"]
        )
        self.assertTrue(
            any_no_ignore["gene_1"] == True
            and any_no_ignore["gene_2"] == False
            and any_no_ignore["gene_3"] == True
            and any_no_ignore["gene_4"] == True
            and any_no_ignore["gene_5"] == True
            and any_no_ignore["gene_6"] is pd.NA
            and any_no_ignore["gene_7"] == True
        )
        self.assertTrue(all(any_no_ignore.index == expected_index))
        self.assertTrue(
            any_ignore_na["gene_1"] == True
            and any_ignore_na["gene_2"] == False
            and any_ignore_na["gene_3"] == True
            and any_ignore_na["gene_4"] == True
            and any_ignore_na["gene_5"] == True
            and any_ignore_na["gene_6"] == False
            and any_ignore_na["gene_7"] == True
        )
        self.assertTrue(all(any_ignore_na.index == expected_index))
        self.assertTrue(
            all_no_ignore["gene_1"] == True
            and all_no_ignore["gene_2"] == False
            and all_no_ignore["gene_3"] == False
            and all_no_ignore["gene_4"] == False
            and all_no_ignore["gene_5"] == False
            and all_no_ignore["gene_6"] == False
            and all_no_ignore["gene_7"] is pd.NA
        )
        self.assertTrue(all(all_no_ignore.index == expected_index))
        self.assertTrue(
            all_ignore_na["gene_1"] == True
            and all_ignore_na["gene_2"] == False
            and all_ignore_na["gene_3"] == False
            and all_ignore_na["gene_4"] == False
            and all_ignore_na["gene_5"] == False
            and all_ignore_na["gene_6"] == False
            and all_ignore_na["gene_7"] == True
        )
        self.assertTrue(all(all_ignore_na.index == expected_index))
        self.assertTrue(
            maj_no_ignore["gene_1"] == True
            and maj_no_ignore["gene_2"] == False
            and maj_no_ignore["gene_3"] == False
            and maj_no_ignore["gene_4"] is pd.NA
            and maj_no_ignore["gene_5"] is pd.NA
            and maj_no_ignore["gene_6"] == False
            and maj_no_ignore["gene_7"] == True
        )
        self.assertTrue(all(maj_no_ignore.index == expected_index))
        self.assertTrue(
            maj_ignore_na["gene_1"] == True
            and maj_ignore_na["gene_2"] == False
            and maj_ignore_na["gene_3"] == False
            and maj_ignore_na["gene_4"] == True
            and maj_ignore_na["gene_5"] == False
            and maj_ignore_na["gene_6"] == False
            and maj_ignore_na["gene_7"] == True
        )
        self.assertTrue(all(maj_ignore_na.index == expected_index))


if __name__ == "__main__":
    unittest.main()
