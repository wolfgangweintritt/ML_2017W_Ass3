#!/usr/bin/env python3
"""Rename the columns of a data set to fit a unified schema"""

import argparse
import csv
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


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
parser.add_argument("--drop", "-d", metavar="COLS", type=str, default="",
                    help="Drop the specified columns from the data set (comma-separated list)")
parser.add_argument("--text-columns", "-t", metavar="COLS", type=str, default="",
                    help="Employ text feature extraction on the specified columns (comma-separated list)")
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
drop_cols = args.drop
text_cols = args.text_columns
try:
    # parse the comma-separated list and discard empty values
    skip_cols = [c for c in skip_cols.split(",") if c != ""]
except:
    skip_cols = []
try:
    drop_cols = [c for c in drop_cols.split(",") if c != ""]
except:
    drop_cols = []
try:
    text_cols = [c for c in text_cols.split(",") if c != ""]
except:
    text_cols = []

# read the data set
dataframe = pd.read_csv(dataset)

# the class name should always be renamed to "Class"
mapping = {class_name: "Class"}

# if given, drop the specified columns
if drop_cols:
    dataframe.drop(columns=drop_cols, inplace=True)

# do the replacement of missing values
for col in dataframe.columns:
    missing_value = "?"

    # if there are no missing values in the column, there's nothing to do
    if missing_value not in dataframe[col]:
        continue

    if dataframe[col].dtype == "O":
        # string handling: use most-used value
        # count the number of occurrences of each value
        val_occs = {v: 0 for v in set(dataframe[col].values)}
        for val in dataframe[col]:
            val_occs[val] += 1

        # find out the most used one
        most_used = None
        most_count = 0
        for (val, cnt) in val_occs.items():
            if cnt >= most_count:
                most_used = val
                most_count = cnt

        dataframe.replace({col: {missing_value: most_used}}, inplace=True)
    else:
        # numeric handling: take mean
        sum_vals = 0
        for val in dataframe[col]:
            sum_vals += val

        mean = sum_vals / len(dataframe[col])
        dataframe.replace({col: {missing_value: mean}}, inplace=True)

# Employ text feature extraction.
removed_text_cols = {c: (dataframe.columns.get_loc(c), dataframe[c]) for c in text_cols}
vect = CountVectorizer(min_df=0., max_df=1.0)
for col in removed_text_cols:
    (idx, values) = removed_text_cols[col]
    # CountVectorizer.fit_transform returns a document-term matrix.
    X = vect.fit_transform(values)

    # insert the new text feature columns at the correct position of the DF
    dataframe = pd.concat([dataframe.loc[:, :col], pd.DataFrame(X.A), dataframe.loc[:, col:]], axis=1)

dataframe.drop(columns=text_cols, inplace=True)

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
        # just plain nominal to ordinal conversion
        for col in dataframe.columns:
            vals = {}
            cur_val = 0

            # dtype("O"): object, which is kind of string
            if dataframe[col].dtype == "O" and col not in skip_cols:
                # sort the lines, such that the same lines are grouped
                rows = sorted(dataframe[col][:])

                old_line = None
                for line in rows:
                    # if the last line was different from the current line,
                    # we have to assign a new integer
                    if old_line is None or old_line != line:
                        vals[line] = cur_val
                        cur_val += 1

                    old_line = line

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

# index = False                  to prevent printing the line number
# quoting = csv.QUOTE_NONNUMERIC to use quotes on non-numeric rows
result = dataframe.to_csv(index=False, quoting=csv.QUOTE_NONNUMERIC)

# print the result to stdout or a file
if out_file is None or out_file == "-":
    try:
        print(result)
    except BrokenPipeError:
        pass
else:
    with open(out_file, "w") as f:
        f.writelines(result)
