import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

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
import comprehensive_functions as comp

# Load data
df = pd.read_csv("Data/cell_cycle_tidied.csv")

# Define features and target
X = df.drop(columns=['phase', 'age', 'PHATE_1', 'PHATE_2'])  # Features
y = df['age']  # Target: age

# define models
DecisionTreeReg = DecisionTreeRegressor(max_depth = 20, 
                                 max_features=150,
                                 random_state=949
                                 )

MLPreg = MLPRegressor(max_iter=200,
                       random_state=949,
                       hidden_layer_sizes = (150,150,150),
                       learning_rate = 'adaptive', 
                       learning_rate_init = 0.001, 
                       alpha = 25)

xgbreg = GradientBoostingRegressor(
        n_estimators=10,       # fixed boosting rounds
        learning_rate=0.5, # hyperparameters from XGB model
        max_depth=5,
        random_state=949
    )

models = [DecisionTreeReg, MLPreg, xgbreg]

J1 = 0
J2 = 1
m_ratio = 0.5
n_ratio = 0.5
B = 5000

mp_ensemble = comp.mp_ensemble(X, y, n_ratio, m_ratio, B, models, adjust_col=None)

feature_groups = comp.make_feature_groups(X, order=2, include_col=None, subset=None)

mp_results = comp.featureInteractions(X, y, mp_ensemble, feature_groups, order=2, type = "regression", bonferroni = True)

np.savez("ensemble_mlp.npz", predictions=mp_ensemble['predictions'], obs=mp_ensemble['mp_observations'], feats=mp_ensemble['mp_features'])

mp_results.to_csv("iloco_ensemble_comprehensvie.csv", index=False)