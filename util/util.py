"""Utility functions shared across several scripts for Machine Learning 2017W"""

import json

class Configuration:
    """Represents a read configuration"""
    def __init__(self, mlp, knn, rf, nb, general):
        self.mlp = mlp
        self.knn = knn
        self.rf = rf
        self.nb = nb

        self.training_split = 0.75
        self.cross_validation = 10
        self.scoring = "accuracy"
        self.target_feature = "Class"

        if "training-split" in general:
            self.training_split = general["training-split"]
        
        if "cross-validation" in general:
            self.cross_validation = general["cross-validation"]

        if "scoring" in general:
            self.scoring = general["scoring"]
        
        if "target-feature" in general:
            self.target_feature = general["target-feature"]


def parse_config_file(file_name):
    """Parse the JSON-formatted configuration file into a Dictionary"""
    with open(file_name) as config_file:
        config = json.load(config_file)
    return config


def read_config(file_name):
    """Read the configuration file and parse it into a Configuration object"""
    cfg = parse_config_file(file_name)
    knn = {}
    mlp = {}
    rf = {}
    nb = {}
    general = {}
    
    if "knn" in cfg:
        knn = cfg["knn"]
    if "mlp" in cfg:
        mlp = cfg["mlp"]
    if "forest" in cfg:
        rf = cfg["forest"]
    if "bayes" in cfg:
        nb = cfg["bayes"]
    if "general" in cfg:
        general = cfg["general"]

    # TODO there should be some input-sanitizing going on here
    return Configuration(mlp, knn, rf, nb, general)


def split_data_set(dataframe, class_name="Class"):
    """Split the data frame into features and class"""
    data = dataframe.drop(columns=[class_name])
    targets = dataframe[class_name].values

    return (data, targets)


def instances_per_class(targets):
    """Calculate the number of instances per class"""
    instances = {c: 0 for c in set(targets)}
    for t in targets:
        instances[t] += 1

    return instances.values()


# if executed as main, do some testing
if __name__ == "__main__":
    cfg = parse_config_file("test.cfg")
    print(cfg)
    print(cfg["test"])
    print(cfg["yes"])
    print(cfg["yes"][0])