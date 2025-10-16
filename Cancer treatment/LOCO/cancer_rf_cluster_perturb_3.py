import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import mode

import sys
sys.path.append("feature_importance")

import locomp
from locomp import *
from locomp.MLmodels import *
from locomp.util_locomp import *
import itertools
import importlib
from sklearn.base import BaseEstimator, RegressorMixin, clone
import itertools
from functools import partial
import multiprocessing as mp
import re

import functions_case as il
import importlib

import functions_tertiary as il3

# Load data
df = pd.read_csv("Cancer/T47D.csv")

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

# Separate features and target
X = df.drop(columns=['phase'])
y = le.fit_transform(df['phase'])
perturbation_col = X.columns.get_loc("Metadata_well")

feature_names = X.columns.tolist()
feature_pairs = list(itertools.combinations(range(X.shape[1]), 2))
feature_triplets = list(itertools.combinations(range(X.shape[1]), 3))
X = X.to_numpy()
#y = y.to_numpy()

def get_mp_metric(X, Y, n_ratio, m_ratio, B, 
                   models, feature_groups, order, type, perturbation_col, alpha, bonferroni = False):
    """
    Computes interaction metrics (iLOCO) for multiple feature pairs using an ensemble model.

    Parameters:
    X (numpy.ndarray): Feature matrix.
    Y (numpy.ndarray): Target variable.
    n_ratio (float): Ratio parameter for ensemble fitting.
    m_ratio (float): Ratio parameter for ensemble fitting.
    B (int): Number of bootstrap samples.
    model (object): Trained model for interaction computation.
    feature_pairs (list of tuples): List of tuples where each tuple contains two feature indices (j1, j2) for the feature pair.

    Returns:
    dict: Dictionary where keys are feature pairs (j1, j2) and values are dictionaries containing:
          - 'iloco': The mean interaction value.
          - 'iloco_max': The maximum interaction value, ensuring non-negative.
          - 'iloco_ratio': The ratio of iloco to the mean of r.
    """
    # Create and fit an ensemble model
    ensemble = il3.Ensemble(models).fit(X, Y, n_ratio, m_ratio, B, perturbation_col = perturbation_col)

    # Compute feature interactions for the given list of feature pairs
    result = il3.featureInteractions(X, Y, n_ratio, m_ratio, B,
                                    models, feature_groups, order, type, perturbation_col, alpha, bonferroni)

    return result

DecisionTreeClass = DecisionTreeClassifier(max_depth = 50, 
                                 max_features=10,
                                 random_state=949
                                 )

J1 = 0
J2 = 1
m_ratio = 0.5
n_ratio = 0.5
B = 5000
models = [DecisionTreeClass]

mp_results = []
mp_ci = []
mp_results = get_mp_metric(X, y, n_ratio, m_ratio, B, models = models, order = 3, type = "classification",
                            perturbation_col=perturbation_col,
                            feature_groups=feature_triplets, alpha = 0.1, bonferroni= True)
mp_scores = [mp_results[(i, j, k)]['iloco'] for (i, j, k) in feature_triplets]
mp_ci = [mp_results[(i, j, k)]["ci"] for (i, j, k) in feature_triplets]

# Generate all pairwise combinations
pairwise_names = list(itertools.combinations(feature_names, 3))
# Convert each tuple to a single string
pairwise_names_str = [' & '.join(pair) for pair in pairwise_names]
data = {'feature': pairwise_names_str, 'scores': mp_scores, 'ci': mp_ci }
df = pd.DataFrame(data)

df.to_csv("perturb_3_decisionTree.csv", index=False)