#!/usr/bin/env python3
"""Rename the columns of a data set to fit a unified schema"""

import argparse
import pandas as pd


# set up the argument parser
descr = "Rename the features of the data set so that they fit a unified schema"
epilog = "2018 - Max Moser"

parser = argparse.ArgumentParser(description=descr, epilog=epilog)
parser.add_argument("--all", "-a", action="store_true", default=False,
                    help="Standardize other feature names too (to V1..Vn)")
parser.add_argument("--output-file", "-o", type=str, default=None,
                    help="Name of the output file - if none is specified, the result will be printed to stdout")
parser.add_argument("dataset", metavar="DATASET", type=str,
                    help="The data set to standardize")
parser.add_argument("classname", metavar="CLASS", type=str,
                    help="Name of the Class (classification target)")

# do the argument parsing
args = parser.parse_args()
rename_all = args.all
dataset = args.dataset
class_name = args.classname
out_file = args.output_file

# read the data set
dataframe = pd.read_csv(dataset)

# the class name should always be renamed to "Class"
mapping = {class_name: "Class"}

if rename_all:
    # if we choose to rename all features, make them to V1,...,Vn
    i = 0
    for col in [c for c in dataframe.columns if c != class_name]:
        i += 1
        new_name = "V%d" % i
        mapping[col] = new_name

# TODO: make string-type features to numeric features (either one-hot encoding or the other one)

# rename the columns of the data set
print(dataframe.columns)
dataframe.rename(index=str, columns=mapping, inplace=True)
print(dataframe.columns)
result = dataframe.to_csv()

# print the result to stdout or a file
if out_file is None or out_file == "-":
    print(result)
else:
    with open(out_file, "w") as f:
        f.writelines(result)
