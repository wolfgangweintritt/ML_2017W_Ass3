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
parser.add_argument("--one-hot", "-1", dest="method", action="store_const",
                    const="one-hot", default="convert",
                    help="If this flag is specified, One-Hot encoding will be used instead of flat conversion to integers")
parser.add_argument("--skip-strings", "-s", action="store_true",
                    default=False, help="Skip the string replacment by numeric values")
parser.add_argument("--skip-columns", "-S", metavar="COLS", type=str, default="",
                    help="Skip the string replacement for the specified columns (comma-separated list)")
parser.add_argument("dataset", metavar="DATASET", type=str,
                    help="The data set to standardize")
parser.add_argument("classname", metavar="CLASS", type=str,
                    help="Name of the Class (classification target)")

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

# read the data set
dataframe = pd.read_csv(dataset)

# the class name should always be renamed to "Class"
mapping = {class_name: "Class"}

# do the string replacement before renaming (b/c skipping columns is name-based)
if not skip_strings:
    if method == "one-hot":
        # save any columns to skip
        cols = {c: (dataframe.columns.get_loc(c), dataframe[c]) for c in skip_cols}
        dataframe.drop(columns=skip_cols, inplace=True)
        
        # get_dummies from pandas does the one-hot encoding for us
        dataframe = pd.get_dummies(dataframe)

        # re-add the skipped columns
        for col in cols:
            (idx, values) = cols[col]
            dataframe.insert(loc=idx, column=col, value=values)
    else:
        for col in dataframe.columns:
            vals = {}
            cur_val = 0

            # dtype("O"): object, which is kind of string
            if dataframe[col].dtype == "O" and col not in skip_cols:
                # it's a sting and we need to convert it
                for (idx, line) in enumerate(dataframe[col]):
                    if line not in vals:
                        vals[line] = cur_val
                        cur_val += 1

                    # replace the strings by numbers
                    dataframe.replace({col: vals}, inplace=True)

# rename the columns
if rename_all:
    # if we choose to rename all features, make them to V1,...,Vn
    i = 0
    for col in [c for c in dataframe.columns if c != class_name]:
        i += 1
        new_name = "V%d" % i
        mapping[col] = new_name

# rename the columns of the data set
dataframe.rename(index=str, columns=mapping, inplace=True)
result = dataframe.to_csv()

# print the result to stdout or a file
if out_file is None or out_file == "-":
    print(result)
else:
    with open(out_file, "w") as f:
        f.writelines(result)
