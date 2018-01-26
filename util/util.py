"""Utility functions shared across several scripts for Machine Learning 2017W"""

import json


def parse_config_file(file_name):
    with open(file_name) as config_file:
        config = json.load(config_file)
    return config


def read_clf_args(file_name):
    cfg = parse_config_file(file_name)
    knn = {}
    mlp = {}
    rf = {}
    nb = {}
    
    if "knn" in cfg:
        knn = cfg["knn"]
    if "mlp" in cfg:
        mlp = cfg["mlp"]
    if "forest" in cfg:
        rf = cfg["forest"]
    if "bayes" in cfg:
        nb = cfg["bayes"]

    # TODO there should be some input-sanitizing going on here
    return (knn, mlp, rf, nb)


# if executed as main, do some testing
if __name__ == "__main__":
    cfg = parse_config_file("test.cfg")
    print(cfg)
    print(cfg["test"])
    print(cfg["yes"])
    print(cfg["yes"][0])