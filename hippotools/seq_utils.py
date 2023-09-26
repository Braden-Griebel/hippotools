"""
Functions for converting between different gene expression units and converting expression into reaction weights
"""
# Standard library imports
from typing import Callable, Union
from warnings import warn

# External imports
import numpy as np
import pandas as pd
from dexom_python import expression2qualitative


def count_to_rpkm(count: pd.DataFrame, feature_length: pd.Series) -> pd.DataFrame:
    """
    Normalize raw count data using RPKM

    :param count: Dataframe containing gene count data, with genes as the columns and samples as the rows
    :type count: pd.DataFrame
    :param feature_length: Series containing the feature length for all the genes
    :type feature_length: pd.Series
    :return: RPKM normalized counts
    :rtype: pd.DataFrame
    """
    # Ensure that the count data frame and feature length series have the same genes
    count_genes = set(count.columns)
    fl_genes = set(feature_length.index)
    if not (count_genes == fl_genes):
        warn(
            "Different genes in count dataframe and feature length series, dropping any not in common"
        )
        genes = count_genes.intersection(fl_genes)
        count = count[genes]
        feature_length = feature_length[genes]
    sum_counts = count.sum(axis=1)
    return count.divide(feature_length, axis=1).divide(sum_counts, axis=0) * 1.0e9


def count_to_fpkm(count: pd.DataFrame, feature_length: pd.Series) -> pd.DataFrame:
    """
    Converts count data to FPKM normalized expression

    :param count: Dataframe containing gene count data, with genes as the columns and samples as the rows. Specifically,
        the count data represents the number of fragments, where a fragment corresponds to a single cDNA molecule, which
        can be represented by a pair of reads from each end.
    :type count: pd.DataFrame
    :param feature_length: Series containing the feature length for all the genes
    :type feature_length: pd.Series
    :return: FPKM normalized counts
    :rtype: pd.DataFrame
    """
    return count_to_rpkm(count, feature_length)


def count_to_tpm(count: pd.DataFrame, feature_length: pd.Series) -> pd.DataFrame:
    """
    Converts count data to TPM normalized expression

    :param count: Dataframe containing gene count data, with genes as the columns and samples as the rows
    :type count: pd.DataFrame
    :param feature_length: Series containing the feature length for all the genes
    :type feature_length: pd.Series
    :return: TPM normalized counts
    :rtype: pd.DataFrame
    """
    # Ensure that the count data frame and feature length series have the same genes
    count_genes = set(count.columns)
    fl_genes = set(feature_length.index)
    if not (count_genes == fl_genes):
        warn(
            "Different genes in count dataframe and feature length series, dropping any not in common"
        )
        genes = count_genes.intersection(fl_genes)
        count = count[genes]
        feature_length = feature_length[genes]
    length_normalized = count.divide(feature_length, axis=1)
    return length_normalized.divide(length_normalized.sum(axis=1), axis=0) * 1.0e6


def count_to_cpm(count: pd.DataFrame) -> pd.DataFrame:
    """
    Converts count data to counts per million

    :param count: Dataframe containing gene count data, with genes as the columns and samples as the rows
    :type count: pd.DataFrame
    :return: CPM normalized counts
    :rtype: pd.DataFrame
    """
    total_reads = count.sum(axis=1)
    per_mil_scale = total_reads / 1e6
    return count.divide(per_mil_scale, axis=0)


def rpkm_to_tpm(rpkm: pd.DataFrame):
    """
    Convert RPKM normalized counts to TPM normalized counts

    :param rpkm: RPKM normalized count data, with genes as columns and samples as rows
    :type rpkm: pd.DataFrame
    :return: TPM normalized counts
    :rtype: pd.DataFrame
    """
    return rpkm.divide(rpkm.sum(axis=1), axis=0) * 1.0e6


def fpkm_to_tpm(fpkm: pd.DataFrame):
    """
    Convert FPKM normalized counts to TPM normalized counts

    :param fpkm: RPKM normalized count data, with genes as columns and samples as rows
    :type fpkm: pd.DataFrame
    :return: TPM normalized counts
    :rtype: pd.DataFrame
    """
    return rpkm_to_tpm(fpkm)


def expression_to_weights(
    gene_expression: pd.DataFrame,
    proportion: Union[float, tuple[float, float]],
    gene_axis: int = 1,
    agg_fun: Callable[[pd.Series], Union[float, int]] = np.median,
) -> pd.Series:
    """
    Function to convert gene expression data into qualitative gene weights (for use in imat and enumeration methods)

    :param gene_expression: Gene expression data, can be a Dataframe with genes as one index and samples as the other,
        or a Series with genes as the index.
    :type gene_expression: Union[pd.DataFrame, pd.Series]
    :param proportion: Proportion to use for trinarizing the data. If a float between 0 and 1, values below that
        percentile will be considered low expression, values whose percentiles are
        above 1-proportion will be considered high expression. If a tuple, the first element will be used as the low
        expression percentile threshold, and the second element will be used as the high expression percentile
        threshold.
    :type proportion: Union[float, tuple[float,float]]
    :param gene_axis: Which axis represents genes in the gene_expression parameter, with 0 being rows and 1 being
        columns
    :type gene_axis: int
    :param agg_fun: Which function to use for combining multiple expression values into 1 which can then be converted
        into the qualitative reaction weight.
    :type agg_fun: Callable[[pd.Series], Union[float,int]]
    :return: Reaction weights
    :rtype: pd.Series
    """
    # If the genes are columns, transpose the dataframe so that they are now
    if gene_axis == 1:
        gene_expression = gene_expression.transpose()
    # If there are multiple columns, they need to be aggregated
    if (not (type(gene_expression) == pd.Series)) and gene_expression.shape[1] > 1:
        gene_expression = gene_expression.aggregate(agg_fun, axis=1)
    # Convert the one column dataframe into a pandas series
    gene_expression = gene_expression.squeeze()
    qual_gene_expression = expression2qualitative(
        gene_expression,
        column_list=None,
        proportion=proportion,
        significant_genes="both",
        save=False,
    )
    return qual_gene_expression
