import numpy as np
from scipy.stats import norm
from sklearn.base import clone
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_rel

def ReLU(arr: np.ndarray) -> np.ndarray:

    """
    Rectified linear unit (ReLU) activation function.
    Generally defined as ReLU(a)=max(0,a).

    Inputs:
      * arr - numpy array representing a column vector.

    Returns:
      * numpy array where for each element e in arr, we have max(0, e).
    """
    return (arr > 0) * arr


def sim_dat_case1(snr, N = 1000, M = 100, signal_features=5,seed=42, interaction = None):
    """
    Generates simulated data for an additive linear model.
    
    Parameters:
    N (int): The number of observations.
    M (int): The number of features.
    signal_features (int): The number of features with a signal.
    seed (int): The random seed for reproducibility.
    
    Returns:
    X (numpy.ndarray): The feature matrix.
    y (numpy.ndarray): The target variable.
    beta (numpy.ndarray): The true regression coefficients.
    """
    #assert interaction in [None, “linear”, “nonlinear”]

    # Set the random seed for reproducibility
    np.random.seed(seed)
    
    # Generate the feature matrix
    X = np.random.normal(loc=0, scale=1, size=(N, M))
    
    # Generate the true regression coefficients
    beta = np.random.normal(loc=2, scale=0.5, size=M)

    #Set non-signal features to zero
    beta[signal_features:] = 0
    # Calculate the target variable
    y = X @ beta + np.random.normal(loc=0, scale=1, size=N)

    if interaction == 'linear':
        y += snr * (X[:, 0] * X[:, 1])
    elif interaction == 'nonlinear':
        y += snr * np.tanh(((X[:, 0] * X[:, 1])))

    return X, y


def buildMP(X, Y, n_ratio, m_ratio):
    """
    Builds a minipatch. Returns:
    - idx_I = the chosen subset of observation indices
    - idx_F = the chosen subset of feature indices
    - x_mp  = the minipatch of observations
    - y_mp  = the minipatch of responses
    """

    N, M = len(X), len(X[0])
    n = int(n_ratio * N)
    m = int(m_ratio * M)
    
    # uniformly sample a subset of observations
    idx_I = np.random.choice(N, size=n, replace=False)
    idx_I.sort()
    # uniformly sample a subset of features
    idx_F = np.random.choice(M, size=m, replace=False)
    idx_F.sort()

    ## record which obs/features are subsampled 
    x_mp = X[np.ix_(idx_I, idx_F)]
    y_mp = Y[np.ix_(idx_I)]
    return idx_I, idx_F, x_mp, y_mp

class Ensemble:
    def __init__(self, model):
        self.base = model

    def fit(self, X, Y, n_ratio, m_ratio, B):
        N, M = X.shape
        self.mp_observations = np.zeros((N, B), dtype=bool)
        self.mp_features = np.zeros((M, B), dtype=bool)
        self.ensemble = [None] * B
        for b in range(B):
            idx_I, idx_F, x_mp, y_mp = buildMP(X, Y, n_ratio, m_ratio)
            self.ensemble[b] = clone(self.base).fit(x_mp, y_mp) 
            self.mp_observations[idx_I, b] = True
            self.mp_features[idx_F, b] = True  
        return self
        
    def predict(self, X):
        predictions = np.empty((len(X), len(self.ensemble)))
        for b, m in enumerate(self.ensemble):
            predictions[:, b] = m.predict(X[:, self.mp_features[:, b]])
        return predictions
    


def predict(X, Y, n_ratio, m_ratio, B, model):
    """
    Fits and predicts models
    """
    N, M = X.shape
    mp_observations = np.zeros((N, B), dtype=bool)
    mp_features = np.zeros((M, B), dtype=bool)
    predictions = np.empty((N, B))
    for b in range(B):
        idx_I, idx_F, x_mp, y_mp = buildMP(X, Y, n_ratio, m_ratio)
        predictions[:, b] = model.fit(x_mp, y_mp).predict(X[:, idx_F])
        mp_observations[idx_I, b] = True
        mp_features[idx_F, b] = True  
    return predictions, mp_observations, mp_features


def computeDeltaCap(Y, j1, j2, predictions, mp_observations, mp_features, 
        metric=np.square):
    """
    TODO: apply bonferroni correction if more than one test is made.
    Computes the squared error vectors from LOCO and LOO predictions.
    """
    
    loo = 1 - mp_observations
    mu_loo = np.sum(predictions * loo, axis=1) / np.sum(loo, axis=1)
    
    loco1 = loo * (1 - mp_features[j1, :])
    mu_loco1 = np.sum(predictions * loco1, axis=1) / np.sum(loco1, axis=1)

    loco2 = loo * (1 - mp_features[j2, :])
    mu_loco2 = np.sum(predictions * loco2, axis=1) / np.sum(loco2, axis=1)

    loco12 = loco1 * loco2
    mu_loco12 = np.sum(predictions * loco12, axis=1) / np.sum(loco12, axis=1)




    residual_loo = metric(Y - mu_loo)
    residual_loco1 = metric(Y - mu_loco1)
    residual_loco2 = metric(Y - mu_loco2)
    residual_loco12 = metric(Y - mu_loco12)

    return residual_loco12, residual_loco1,  residual_loco2, residual_loo



def getCI(delta_cap, alpha=0.1):
    sigma = np.std(delta_cap, ddof=1)
    ci = norm.ppf(1 - alpha / 2) * sigma / np.sqrt(len(delta_cap))
    return ci


#def featureInteractions(X, Y, n_ratio, m_ratio, B, model, feature_pairs, alpha=0.1, bonferroni=False):
    """
    Computes interaction metrics (iLOCO) for multiple feature pairs.

    Parameters:
    - X, Y: Feature matrix and target variable.
    - n_ratio, m_ratio: Ratios for minipatching.
    - B: Number of bootstraps.
    - model: Base model used.
    - feature_pairs: List of (j1, j2) feature index tuples.
    - alpha: Significance level for confidence interval.
    - bonferroni: If True, applies Bonferroni correction.

    Returns:
    - dict mapping (j1, j2) to metrics: iloco, iloco_max, iloco_ratio, ci
    """
    predictions, mp_observations, mp_features = predict(X, Y, n_ratio, m_ratio, B, model)
    results = {}

    # Adjust alpha if Bonferroni is requested
    if bonferroni:
        adjusted_alpha = alpha / len(feature_pairs)
    else:
        adjusted_alpha = alpha

    for (j1, j2) in feature_pairs:
        r12, r1, r2, r = computeDeltaCap(Y, j1, j2, predictions, mp_observations, mp_features)
        dc = r1 + r2 - r12 - r
        iloco = np.mean(dc)
        iloco_max = max(0, iloco)
        iloco_ratio = iloco / np.mean(r)
        dc = r1 - r
        ci = getCI(dc, alpha=adjusted_alpha)

        results[(j1, j2)] = {
            'iloco': iloco,
            'iloco_max': iloco_max,
            'iloco_ratio': iloco_ratio,
            'ci': ci
        }

    return results

import multiprocessing as mp
from functools import partial
from scipy.stats import norm
import numpy as np

def compute_interaction_for_pair(j1, j2, Y, predictions, mp_observations, mp_features, alpha):
    r12, r1, r2, r = computeDeltaCap(Y, j1, j2, predictions, mp_observations, mp_features)
    dc = r1 + r2 - r12 - r
    iloco = np.mean(dc)
    iloco_max = max(0, iloco)
    iloco_ratio = iloco / np.mean(r)
    dc_ci = r1 - r
    ci = getCI(dc_ci, alpha=alpha)

    return ((j1, j2), {
        'iloco': iloco,
        'iloco_max': iloco_max,
        'iloco_ratio': iloco_ratio,
        'ci': ci
    })

def featureInteractions(X, Y, n_ratio, m_ratio, B, model, feature_pairs, alpha=0.1, bonferroni=False, n_jobs=None):
    """
    Parallelized computation of interaction metrics (iLOCO) for feature pairs.

    Parameters:
    - n_jobs: number of processes (default: all cores)

    Returns:
    - dict: results for each (j1, j2) pair
    """
    predictions, mp_observations, mp_features = predict(X, Y, n_ratio, m_ratio, B, model)

    # Adjust alpha for Bonferroni correction
    adjusted_alpha = alpha / len(feature_pairs) if bonferroni else alpha

    # Use partial to pass fixed args
    func = partial(
        compute_interaction_for_pair,
        Y=Y,
        predictions=predictions,
        mp_observations=mp_observations,
        mp_features=mp_features,
        alpha=adjusted_alpha
    )

    with mp.Pool(processes=n_jobs) as pool:
        results_list = pool.starmap(func, feature_pairs)

    return dict(results_list)
