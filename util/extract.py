#!/usr/bin/env python3
"""Extract features out of datasets"""

import sys
import argparse
import pandas as pd
import numpy as np
from scipy import stats

DATASET_PATH = "../datasets/testing/test.csv"

def main():
    # read the data set
    dataframe = pd.read_csv(DATASET_PATH)

    print(extractFeatures(dataframe))


# returns a dictonary which contains all the extracted features out of the given dataset
def extractFeatures(dataset):

    output_dic = {}

    # number of instances
    output_dic["#instances"] = dataset.shape[0]

    # number of features
    output_dic["#features"] = dataset.shape[1]

    # number of classes
    output_dic["#classes to predict"] = dataset["ID"].unique().size

    # variance
    feature_variances = np.var(dataset)
    output_dic["min variance"] = feature_variances.min()
    output_dic["max variance"] = feature_variances.max()
    output_dic["mean variance"] = feature_variances.mean()


    # correlation
    cor_matrix = np.corrcoef(dataset.transpose())
    # diagonal is not interesting since it determines the correlation to itself --> remove the diagonal
    cor_matrix = cor_matrix[~np.eye(cor_matrix.shape[0],dtype=bool)].reshape(cor_matrix.shape[0],-1)

    output_dic["min correlation"] = cor_matrix.min()
    output_dic["max correlation"] = cor_matrix.max()
    output_dic["mean correlation"] = cor_matrix.mean()

    # Scale between features, determines the freature with the minimal and maximal range and computes the relative difference of min/max
    ranges = dataset.apply(lambda x: x.max()-x.min())

    relative_scale_diff = ranges.min() / ranges.max()
    output_dic["relative scale diff"] = relative_scale_diff

    # Normality Test
    '''
    statistic : float or array
    s^2 + k^2, where s is the z-score returned by skewtest and 
    k is the z-score returned by kurtosistest.
    
    pvalue : float or array
    A 2-sided chi squared probability for the hypothesis test.
    
    If the p-val is very small, 
    it means it is unlikely that the data came from a normal distribution.
    https://stackoverflow.com/questions/12838993/scipy-normaltest-how-is-it-used
    '''
    normality_test = dataset.apply(lambda x: stats.normaltest(x))
    norm_p_value = pd.Series([x[1] for x in normality_test])

    output_dic["min normality test"] = norm_p_value.min()
    output_dic["max normality test"] = norm_p_value.max()
    output_dic["mean normality test"] = norm_p_value.mean()

    # randomness test
    '''
    D : float
    
    KS test statistic, either D, D+ or D-.
    
    p-value : float
    
    One-tailed or two-tailed p-value.
    '''
    randomness_test = dataset.apply(lambda x: stats.kstest(x, 'uniform'))
    rand_p_value = pd.Series([x[1] for x in randomness_test])

    output_dic["min randomness test"] = rand_p_value.min()
    output_dic["max randomness test"] = rand_p_value.max()
    output_dic["mean randomness test"] = rand_p_value.mean()

    return output_dic

if __name__ == '__main__':
    main()
