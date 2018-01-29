#!/usr/bin/env python

import argparse
import os.path
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from util.util import read_config, instances_per_class, split_data_set


# set up the argument parser
descr = "Perform basic classification on the meta-learning data set and print the results."
epilog = "2018 - Max Moser"

parser = argparse.ArgumentParser(description=descr, epilog=epilog)
parser.add_argument("dataset", metavar="DATASET", type=str, default=None,
                    help="The dataset on which to perform meta-learning")
parser.add_argument("--no-cross-val", "-n", action="store_true", default=False,
                    help="Do standard training/test split instead of cross-validation")
parser.add_argument("--config", "-c", metavar="CONFIG", type=str, default=None,
                    help="The configuration file to use for the parameters of the classifiers")

# do the argument parsing
args = parser.parse_args()
dataset = args.dataset
config = args.config
no_cv = args.no_cross_val

if not os.path.isfile(dataset):
    print("Error: No such data set: %s" % dataset)
    exit(1)

# read the config
knn_args = {}
mlp_args = {}
rf_args = {}
nb_args = {}
cross_val = 10
split = 0.75
scoring = "accuracy"
class_name = "Class"
if config is not None and os.path.isfile(config):
    cfg = read_config(config)
    knn_args = cfg.knn
    mlp_args = cfg.mlp
    rf_args = cfg.rf
    nb_args = cfg.nb
    cross_val = cfg.cross_validation
    scoring = cfg.scoring
    split = cfg.training_split
    class_name = cfg.target_feature

# read the data set
data = pd.read_csv(dataset)
y = data[class_name].values
X = data.drop(columns=[class_name])
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=(1 - split))

knn_args["n_jobs"] = -1
rf_args["n_jobs"] = -1
# mlp does not accept the argument 'n_jobs' :(

knn = KNeighborsClassifier(**knn_args)
mlp = MLPClassifier(**mlp_args)
rf = RandomForestClassifier(**rf_args)

classifiers = {"k-Nearest Neighbour": knn,
               "Multi-Layer Perceptron": mlp,
               "Random Forest": rf}

for name, clf in classifiers.items():

    print("> %s" % name)
    print()

    if no_cv:
        # standard training/test split stuff
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

    else:
        # cross-validation stuff
        (set_data, set_targets) = split_data_set(data, class_name)
        max_cv = min(instances_per_class(set_targets))
        max_cv = min(max_cv, cross_val)
        scores = cross_val_score(clf, set_data, set_targets, cv=max_cv, n_jobs=-1, scoring=scoring)
        score = scores.mean()

        print("%d-fold Cross-Validation" % max_cv)
        print("Cross validation scores: %s" % scores)
        print("Mean:                    %s" % score)

    print()
    print("=" * 80)