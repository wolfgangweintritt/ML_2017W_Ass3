# Machine Learning 2017W, Assignment 3: Meta-Learning

## Group Members

* Georg Faustmann (01326020)
* Maximilian Moser (01326252)
* Wolfgang Weintritt (01327191)

## Chosen Data Sets

* abalone (http://archive.ics.uci.edu/ml/datasets/Abalone)
* adult (http://archive.ics.uci.edu/ml/datasets/Adult)
* agaricus lepiota: mushroom (https://archive.ics.uci.edu/ml/datasets/mushroom)
* breat cancer wisconsin (https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(original))
* car evaluation (https://archive.ics.uci.edu/ml/datasets/car+evaluation)
* chronic kidney disease (http://archive.ics.uci.edu/ml/datasets/Chronic_Kidney_Disease)
* connect 4 (https://archive.ics.uci.edu/ml/datasets/Connect-4)
* banknote authentication (https://archive.ics.uci.edu/ml/datasets/banknote+authentication)
* dota 2 game results (http://archive.ics.uci.edu/ml/datasets/Dota2+Games+Results)
* heart disease (https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
* congressional voting (https://archive.ics.uci.edu/ml/datasets/congressional+voting+records)
* leaf (https://archive.ics.uci.edu/ml/datasets/leaf)
* letter recognition (https://archive.ics.uci.edu/ml/datasets/Letter+Recognition)
* skillcraft 1 (https://archive.ics.uci.edu/ml/datasets/skillcraft1+master+table+dataset)
* student performance (https://archive.ics.uci.edu/ml/datasets/student+performance)
* wine quality (http://archive.ics.uci.edu/ml/datasets/Wine+Quality)
* youtube spam collection (https://archive.ics.uci.edu/ml/datasets/YouTube+Spam+Collection)

## Extracted Data Set Features

* Number of features / attributes
* Number of instances
* Number of classes
* Mean value of variances of all classes
* Min variance of all features
* Max variance of all features
* Min correlation between features
* Max correlation between features
* Mean correlation between features
* Scale between features
* Min normality test per attribute
* Max normality test per attribute
* Mean normality test per attribute
* Min randomness test per attribute
* Max randomness test per attribute
* Mean randomness test per attribute

## Classifiers

* k-Nearest Neighbour
* Multi-Layer Perceptron
* Random Forest
* Naive Bayes

## Scripts

* `eval-sets.py`: Reads through all data sets in a path and evaluates the above set of classifiers on each data set. Then proceeds to extract all features from the data set and store the extracted features along with the best-performing classifier per data set in a new CSV file.
* `unify.py`: Brings the data sets into a unified format (makes sure that there is one 'Class' attribute and string-valued features are converted to numeric features - either per One-Hot encoding or just conversion to integers).

Note that for `eval-sets.py`, a configuration file can be specified optionally which should contain initialisation parameters for the classifiers in JSON-format. An example for such a configuration can be seen in the file `configuration.cfg`.

## Requirements

* Python Packages:
    * pandas
    * sklearn
    * scipy
    * numpy
