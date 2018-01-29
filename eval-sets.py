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
from util.extract import extractFeatures
from util.util import read_config, instances_per_class, split_data_set



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
        debug("Reading file: %s" % f)
        basename = os.path.basename(f)
        dataset = pd.read_csv(f)
        datasets[basename] = dataset

    return datasets


# set up the argument parser
descr = "Run various classifiers over the data sets in a specified directory and " \
        "check which one yields the best performance per data set. " \
        "Expects unified format of data sets."
epilog = "2018 - Max Moser"

parser = argparse.ArgumentParser(description=descr, epilog=epilog)
parser.add_argument("path", metavar="PATH", type=type_path,
                    nargs="?", default=".",
                    help="The path in which to look for data sets, defaults to working directory")
parser.add_argument("--output", "-o", metavar="OUTPUT-FILE", type=str, default="meta-learning.csv",
                    help="The file in which to store the meta-learning data set")
parser.add_argument("--config", "-c", metavar="CONFIG", type=str, default=None,
                    help="The configuration file to use for the parameters of the classifiers")

# do the argument parsing and initialize options
args = parser.parse_args()
path = args.path
out_file = args.output
config_file = args.config
debug("Using path: %s" % path)

# read the config
knn_args = {}
mlp_args = {}
rf_args = {}
nb_args = {}
cross_val = 10
scoring = "accuracy"
class_name = "Class"
if config_file is not None and os.path.isfile(config_file):
    cfg = read_config(config_file)
    knn_args = cfg.knn
    mlp_args = cfg.mlp
    rf_args = cfg.rf
    nb_args = cfg.nb
    cross_val = cfg.cross_validation
    scoring = cfg.scoring
    class_name = cfg.target_feature

datasets = datasets_in_path(path)
for name, dataset in datasets.copy().items():
    if class_name not in dataset.columns:
        # skip every data set that doesn't fit our schema
        print("Warn: Skipping data set '%s' (missing feature: '%s')" % (name, class_name))
        del datasets[name]

# if we specified an output-file (i.e. the file in which the extracted features, etc. are stored)
# then we don't want to use this one as input
if out_file in datasets:
    del datasets[out_file]

debug(datasets.keys())

# set up the classifiers and create Dictionary that assigns them a nice name
knn = KNeighborsClassifier(**knn_args)
mlp = MLPClassifier(**mlp_args)
rf = RandomForestClassifier(**rf_args)
nb = GaussianNB(**nb_args)
classifiers = {"K-Nearest Neighbour": knn,
               "Multi-Layer Perceptron": mlp,
               "Random Forest": rf,
               "Naive Bayes": nb}


meta_learning_dataset = pd.DataFrame()
best_classifiers = {}
for data_name, dataframe in datasets.items():
    print("Data Set: %s" % data_name)
    meta_features = extractFeatures(dataframe)

    # split the data set into features and class
    # and check if we have enough instances per class to do 10-fold CV
    # (if not, lower k)
    (set_data, set_targets) = split_data_set(dataframe, class_name)
    max_cv = min(instances_per_class(set_targets))
    max_cv = min(max_cv, cross_val)

    best = None
    best_performance = 0

    print("Using %d-fold cross-validation" % max_cv)
    print()
    for clf_name, clf in classifiers.items():
        print("> %s" % clf_name)
        debug("> %s" % clf)

        scores = cross_val_score(clf, set_data, set_targets, cv=max_cv, n_jobs=-1, scoring=scoring)
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

    meta_features[class_name] = best

    meta_features_series = pd.Series(meta_features, name=data_name)
    meta_learning_dataset = meta_learning_dataset.append(meta_features_series)

print()
print("Summary, Best Classifiers:")
for (dataset_name, best_classifier) in best_classifiers.items():
    print("> %s: %s" % (dataset_name, best_classifier))

meta_learning_dataset.to_csv(out_file, index=False, quoting=csv.QUOTE_NONNUMERIC)
