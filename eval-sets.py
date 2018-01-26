#!/usr/bin/env python3
"""Let a bunch of classifiers run over all data sets (*.csv) in a specified path"""

import argparse
import glob
import csv
import os.path
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from util.util import read_clf_args


dbg = False


def debug(*args, **kwargs):
    if dbg:
        print(*args, **kwargs)


def type_path(path):
    """Make the argument into an absolute path for argparse"""
    try:
        path = os.path.expanduser(path)
        path = os.path.abspath(path)
        return path
    except:
        raise argparse.ArgumentTypeError("Could not parse the folder: %s" % path)


def datasets_in_path(path):
    """Read the data sets (*.csv) in a path"""
    datasets = {}
    files = glob.glob(os.path.join(path, "*.csv"))
    for f in files:
        basename = os.path.basename(f)
        dataset = pd.read_csv(f)
        datasets[basename] = dataset

    return datasets


def split_data_set(dataframe):
    """Split the data frame into features and class"""
    data = dataframe.drop(columns=["Class"])
    targets = dataframe["Class"].values

    return (data, targets)


def instances_per_class(targets):
    """Calculate the number of instances per class"""
    instances = {c: 0 for c in set(targets)}
    for t in targets:
        instances[t] += 1
    
    return instances.values()


# set up the argument parser
descr = "Run various classifiers over the data sets in a specified directory and " \
        "check which one yields the best performance per data set. " \
        "Expects unified format of data sets."
epilog = "2018 - Max Moser"

parser = argparse.ArgumentParser(description=descr, epilog=epilog)
parser.add_argument("path", metavar="PATH", type=type_path,
                    nargs="?", default=".",
                    help="The path in which to look for data sets, defaults to working directory")
parser.add_argument("--append", "-a", metavar="DATASET", type=str, default=None,
                    help="The data set consisting of extracted features (aka meta-learning data set)" \
                    " to which the results should be appended as a new 'Class' feature")
parser.add_argument("--config", "-c", metavar="CONFIG", type=str, default=None,
                    help="The configuration file to use for the parameters of the classifiers")

# do the argument parsing and initialize options
args = parser.parse_args()
path = args.path
append_file = args.append
config_file = args.config
debug("Using path: %s" % path)

datasets = datasets_in_path(path)
for name, dataset in datasets.copy().items():
    if "Class" not in dataset.columns:
        # skip every data set that doesn't fit our schema
        print("Warn: Skipping data set '%s' (missing feature: 'Class')" % name)
        del datasets[name]

# if we specified an append-file (i.e. the meta-learning data set where to append the results)
# then we don't want to use this one as input
if append_file in datasets:
    del datasets[append_file]

debug(datasets.keys())

knn_args = {}
mlp_args = {}
rf_args = {}
nb_args = {}
if config_file is not None and os.path.isfile(config_file):
    (knn_args, mlp_args, rf_args, nb_args) = read_clf_args(config_file)

# set up the classifiers and create Dictionary that assigns them a nice name
knn = KNeighborsClassifier(**knn_args)
mlp = MLPClassifier(**mlp_args)
rf = RandomForestClassifier(**rf_args)
nb = GaussianNB(**nb_args)
classifiers = {"K-Nearest Neighbour": knn,
               "Multi-Layer Perceptron": mlp,
               "Random Forest": rf,
               "Naive Bayes": nb}

best_classifiers = {}
for data_name, dataframe in datasets.items():
    # split the data set into features and class
    # and check if we have enough instances per class to do 10-fold CV
    # (if not, lower k)
    (set_data, set_targets) = split_data_set(dataframe)
    max_cv = min(instances_per_class(set_targets))
    max_cv = min(max_cv, 10)

    best = None
    best_performance = 0

    print("Data Set: %s" % data_name)
    print("Using %d-fold cross-validation" % max_cv)
    print()
    for clf_name, clf in classifiers.items():
        print("> %s" % clf_name)
        debug("> %s" % clf)

        scores = cross_val_score(clf, set_data, set_targets, cv=max_cv, n_jobs=-1, scoring="accuracy")
        # scoring via "f1" causes problems when there are no samples predicted for a class
        # thus, we just take the default "scoring" method of the classifier -> TODO find something better
        
        # as final performance, we just take the mean of the performances
        performance = scores.mean()
        if performance >= best_performance:
            best = clf_name
            best_performance = performance
        
        print("Performances: %s" % scores)
        print("Mean:         %s" % performance)
        print()

    best_classifiers[data_name] = best
    print("Best Classifier: %s" % best)
    print()

    print("=" * 80)

print()
print("Summary, Best Classifiers:")
for (dataset_name, best_classifier) in best_classifiers.items():
    print("> %s: %s" % (dataset_name, best_classifier))

if append_file is not None:
    if os.path.isfile(append_file):
        # requires an actual meta-learning dataset with the exact same amount of rows
        # as there have been data sets
        append_dataset = pd.read_csv(append_file)
        values = [v for v in best_classifiers.values()]
        last_idx = len(append_dataset.columns)
        append_dataset.insert(loc=last_idx, column="Class", value=values)
        append_dataset.to_csv(append_file, index=False, quoting=csv.QUOTE_NONNUMERIC)
    else:
        print()
        print("Error: Specified append-file '%s' does not exist!" % append_file)
        exit(1)