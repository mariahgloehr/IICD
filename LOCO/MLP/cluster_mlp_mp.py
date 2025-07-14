import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import root_mean_squared_error

# Load data
df = pd.read_csv("Data/cell_cycle_tidied.csv")

# Define features and target
X = df.drop(columns=['phase', 'age', 'PHATE_1', 'PHATE_2'])  # Features
y = df['age']  # Target: age

feature_names = X.columns.tolist()
X = X.to_numpy()
y = y.to_numpy()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=949
)

import sys
sys.path.append("/Users/mariahloehr/IICD/IICD/feature_importance")

import locomp
from locomp import *
from locomp.MLmodels import *
import itertools
import importlib
from sklearn.base import BaseEstimator, RegressorMixin, clone
import itertools
from functools import partial
import multiprocessing as mp
import re

# define fit_func
def MLPreg(X,Y,X1):
    mlp = MLPRegressor(max_iter=200,
                       random_state=949,
                       hidden_layer_sizes = (60,60,60),
                       learning_rate = 'adaptive', 
                       learning_rate_init = 0.001, 
                       alpha = 25).fit(X,Y)
    return mlp.predict(X1)

J1 = 0
J2 = 1
m_ratio = 0.8
n_ratio = 0.9
B = 5000
fit_func = MLPreg

x=LOCOMPReg(X_train,y_train,n_ratio,m_ratio,B,fit_func, selected_features=[],alpha=0.1,bonf=True)
x.run_loco()

ci_df = pd.DataFrame(x.loco_ci)
ci_df = ci_df.rename(columns={2: 'lower_bound', 3: 'upper_bound', 4: 'score'})
ci_df['feature_name'] = feature_names

ci_df.to_csv('mlp_loco_importances.csv', index=False)

