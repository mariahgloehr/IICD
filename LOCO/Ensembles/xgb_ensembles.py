import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
import itertools

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

xgbreg = GradientBoostingRegressor(
        n_estimators=10,       # fixed boosting rounds
        learning_rate=0.5, # hyperparameters from XGB model
        max_depth=5,
        random_state=949
    )

J1 = 0
J2 = 1
m_ratio = 0.6
n_ratio = 0.6
B = 5000

pred_xgb, obs_xgb, feats_xgb = predict(X, y, n_ratio, m_ratio, B, model = xgbreg)

np.savez("ensemble_xgb.npz", predictions=pred_xgb, obs=obs_xgb, feats=feats_xgb)