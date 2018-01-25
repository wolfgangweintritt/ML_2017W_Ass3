#!/usr/bin/env python3
"""Let a bunch of classifiers run over all data sets (*.csv) in a specified path"""

import argparse
import glob
import os.path
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score


dbg = True


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

# do the argument parsing and initialize options
args = parser.parse_args()
path = args.path
debug("Using path: %s" % path)

datasets = datasets_in_path(path)
for name, dataset in datasets.copy().items():
    if "Class" not in dataset.columns:
        # skip every data set that doesn't fit our schema
        print("Warn: Skipping data set '%s' (missing feature: 'Class')" % name)
        del datasets[name]

debug(datasets.keys())

# set up the classifiers and create Dictionary that assigns them a nice name
# TODO optionally read different parameters from somewhere (config file?) and apply them
knn = KNeighborsClassifier()
mlp = MLPClassifier()
rf = RandomForestClassifier()
nb = GaussianNB()
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
