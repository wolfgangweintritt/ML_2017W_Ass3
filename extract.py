#!/usr/bin/env python3
"""Extract features out of datasets"""

import argparse
import pandas as pd
import numpy as np
from scipy import stats


# set up the argument parser
descr = "Extract features out of datasets"
epilog = "2018 - Georg Faustmann"

parser = argparse.ArgumentParser(description=descr, epilog=epilog)
parser.add_argument("--all", "-a", action="store_true", default=False,
                    help="Standardize other feature names too (to V1..Vn)")
parser.add_argument("--output-file", "-o", type=str, default=None,
                    help="Name of the output file - if none is specified, the result will be printed to stdout")
parser.add_argument("--one-hot", "-1", dest="method", action="store_const",
                    const="one-hot", default="convert",
                    help="If this flag is specified, One-Hot encoding will be used instead of flat conversion to integers")
parser.add_argument("--skip-strings", "-s", action="store_true",
                    default=False, help="Skip the string replacment by numeric values")
parser.add_argument("--skip-columns", "-S", metavar="COLS", type=str, default="",
                    help="Skip the string replacement for the specified columns (comma-separated list)")
# parser.add_argument("dataset", metavar="DATASET", type=str,
#                    help="The data set to standardize")

'''
# do the argument parsing and initialize options
args = parser.parse_args()
rename_all = args.all
dataset = args.dataset
class_name = args.classname
out_file = args.output_file
method = args.method
skip_strings = args.skip_strings
skip_cols = args.skip_columns
try:
    # parse the comma-separated list and discard empty values
    skip_cols = [c for c in skip_cols.split(",") if c != ""]
except:
    skip_cols = []
'''


# read the data set
# dataframe = pd.read_csv(dataset)
dataframe = pd.read_csv("./datasets/testing/test.csv")

# number of instances
dataframe.shape[0]

# number of features
dataframe.shape[1]

# number of classes
dataframe["ID"].unique().size

# variance
for name,col in dataframe.items():
    print("-------")
    print(np.var(col))

print(np.var(dataframe))

# correlation
print(dataframe.transpose())
cor_matrix = np.corrcoef(dataframe.transpose())
# diagonal is not interesting since it determines the correlation to itself
cor_matrix = cor_matrix[~np.eye(cor_matrix.shape[0],dtype=bool)].reshape(cor_matrix.shape[0],-1)
print(cor_matrix)

cor_matrix.min()
cor_matrix.max()
cor_matrix.mean()

# Scale between features
delta = dataframe.apply(lambda x: x.max()-x.min())
print(dataframe.apply(lambda x: x.max()-x.min()))
relative_diff = delta.min() / delta.max()

# Normality Test
'''
statistic : float or array
s^2 + k^2, where s is the z-score returned by skewtest and 
k is the z-score returned by kurtosistest.

pvalue : float or array
A 2-sided chi squared probability for the hypothesis test.
'''
k2, p = stats.normaltest(dataframe)

# randomness test
'''
D : float

KS test statistic, either D, D+ or D-.

p-value : float

One-tailed or two-tailed p-value.
'''
print(dataframe.apply(lambda x: stats.kstest(x, 'uniform')))