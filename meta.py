#!/usr/bin/env python

import argparse
import os.path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from util.util import read_clf_args


# set up the argument parser
descr = "Perform basic classification on the meta-learning data set and print the results."
epilog = "2018 - Max Moser"

parser = argparse.ArgumentParser(description=descr, epilog=epilog)
parser.add_argument("dataset", metavar="DATASET", type=str, default=None,
                    help="The dataset on which to perform meta-learning")
parser.add_argument("--config", "-c", metavar="CONFIG", type=str, default=None,
                    help="The configuration file to use for the parameters of the classifiers")

# do the argument parsing
args = parser.parse_args()
dataset = args.dataset
config = args.config

if not os.path.isfile(dataset):
    print("Error: No such data set: %s" % dataset)
    exit(1)

# read the data set
data = pd.read_csv(dataset)
class_name = "Class" # "best classifier"
y = data[class_name].values
X = data.drop(columns=[class_name])
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.25)

# set up the classifiers
knn_args = {}
mlp_args = {}
rf_args = {}
if config is not None and os.path.isfile(config):
    (knn_args, mlp_args, rf_args, nb_args) = read_clf_args(config)

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
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("> %s" % name)
    print()
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print()
    print("=" * 80)
