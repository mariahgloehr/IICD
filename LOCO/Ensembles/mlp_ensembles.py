import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

import sys
sys.path.append("/Users/mariahloehr/IICD/IICD/feature_importance")

import locomp
from locomp import *
from locomp.MLmodels import *
import importlib
from sklearn.base import BaseEstimator, RegressorMixin, clone
import itertools
from functools import partial
import multiprocessing as mp
import re
import functions_case as il
import importlib

# Load data
df = pd.read_csv("/Users/mariahloehr/IICD/IICD/Data/cell_cycle_tidied.csv")

# Define features and target
X = df.drop(columns=['phase', 'age', 'PHATE_1', 'PHATE_2'])  # Features
y = df['age']  # Target: age

feature_names = X.columns.tolist()
feature_pairs = list(itertools.combinations(range(X.shape[1]), 2))
X = X.to_numpy()
y = y.to_numpy()

# define fit_func
MLPreg = MLPRegressor(max_iter=200,
                       random_state=949,
                       hidden_layer_sizes = (150,150,150),
                       learning_rate = 'adaptive', 
                       learning_rate_init = 0.001, 
                       alpha = 25)

J1 = 0
J2 = 1
m_ratio = 0.6
n_ratio = 0.6
B = 5000

pred_mlp, obs_mlp, feats_mlp = il.predict(X, y, n_ratio, m_ratio, B, model = MLPreg)

np.savez("ensemble_mlp.npz", predictions=pred_mlp, obs=obs_mlp, feats=feats_mlp)