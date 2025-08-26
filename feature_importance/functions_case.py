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

def computeDeltaCap_xent(Y,
                         j1, j2,
                         predictions,
                         mp_observations,
                         mp_features,
                         eps: float = 1e-12):
    """
    Cross-entropy (log loss) version of computeDeltaCap that supports:
    - Binary or multiclass
    - Predictions as probabilities OR hard labels per patch
    Parameters
    ----------
    Y : array-like
    Labels. Binary: shape (n,) in {0,1}. Multiclass: (n,) ints 0..K-1 or one-hot (n,K).
    j1, j2 : int
    Feature indices to test for LOCO.
    predictions : np.ndarray
    One of:
    - (n,B) floats in [0,1]: binary probabilities P(y=1) per patch
    - (n,B) ints in {0,1}: binary hard labels per patch
    - (n,B,K) floats: multiclass probs per patch (sum=1 along K)
    - (n,B) ints 0..K-1: multiclass hard labels per patch
    mp_observations : (n,B) bool/int
    1 if obs used in patch, 0 otherwise.
    mp_features : (p,B) bool/int
    1 if feature used in patch, 0 otherwise.
    eps : float
    Numerical stability for logs.
    Returns
    -------
    residual_loco12, residual_loco1, residual_loco2, residual_loo : np.ndarray (length n)
    """
    Y = np.asarray(Y)
    # ---------- infer K and convert Y to one-hot ----------
    def to_one_hot(y, K=None):
        y = np.asarray(y)
        if y.ndim == 2: # already one-hot
            return y.astype(float), y.shape[1]
        if K is None:
            K = int(np.nanmax(y)) + 1
        Yk = np.zeros((y.shape[0], K), dtype=float)
        Yk[np.arange(y.shape[0]), y.astype(int)] = 1.0
        return Yk, K
   
    # ---------- unify predictions to (n,B,K) "per-patch" probs -----------
    P = np.asarray(predictions)
    n, B = mp_observations.shape

    if P.ndim == 3:
        # (n,B,K) probs
        nP, BP, K = P.shape
        assert nP == n and BP == B
        preds_k = P
        # Y one-hot
        if Y.ndim == 2:
            Yk, K_y = Y.astype(float), Y.shape[1]
        else:
            Yk, K_y = to_one_hot(Y, K=K)
        assert K_y == K, "Label classes and prediction classes mismatch."
    
    elif P.ndim == 2:
        nP, BP = P.shape
        assert nP == n and BP == B

        # Distinguish binary probs vs hard labels
        is_int_like = np.all(np.equal(P, np.round(P)))
        if not is_int_like:
        # assume binary *probabilities* (n,B) in [0,1]
            p1 = P
            p0 = 1.0 - p1
            preds_k = np.stack([p0, p1], axis=-1) # (n,B,2)
        if Y.ndim == 2:
            Yk, K = Y.astype(float), Y.shape[1]
            assert K == 2, "Binary probs given but Y has !=2 classes."
        else:
        # Y in {0,1}
            Yk = np.stack([1 - Y, Y], axis=-1).astype(float)
        K = 2

    else:
    # hard labels per patch: could be binary {0,1} or multiclass {0..K-1}
        max_label = int(P.max())
        K = max(2, max_label + 1)
    # one-hot per patch, then average later to get empirical probs
        preds_k = np.zeros((n, B, K), dtype=float)
        rows = np.repeat(np.arange(n), B)
        cols = P.astype(int).ravel()
        preds_k.reshape(-1, K)[rows * 1 + 0, 0] # just to touch array (avoid linter warning)
        preds_k[np.arange(n)[:, None], np.arange(B)[None, :], P.astype(int)] = 1.0

        # Y to one-hot with same K
        if Y.ndim == 2:
            Yk, K_y = Y.astype(float), Y.shape[1]
            assert K_y == K, "Hard-label preds imply K={}, but Y has K={}".format(K, K_y)
        else:
            Yk, _ = to_one_hot(Y, K=K)

    #else:
    #    raise ValueError("`predictions` must be 2D (n,B) or 3D (n,B,K).")
    
    # ---------- helper: masked mean over patches -> per-row probs (n,K) ----
    def masked_mean_probs(preds_k, mask_bool):
        w = mask_bool.astype(float) # (n,B)
        denom = w.sum(axis=1, keepdims=True) # (n,1)
        denom[denom == 0] = np.nan # mark rows with no kept patches
        num = np.nansum(preds_k * w[..., None], axis=1) # (n,K)

        return num / denom # (n,K), NaNs if denom==0
        
    # --------------- build masks (same logic as your regression fn) ----------
    loo = 1 - mp_observations
    mu_loo = masked_mean_probs(preds_k, loo)
    loco1_mask = loo * (1 - mp_features[j1, :])
    mu_loco1 = masked_mean_probs(preds_k, loco1_mask)
    loco2_mask = loo * (1 - mp_features[j2, :])
    mu_loco2 = masked_mean_probs(preds_k, loco2_mask)
    loco12_mask = loco1_mask * loco2_mask
    mu_loco12 = masked_mean_probs(preds_k, loco12_mask)

    # ---------------- cross-entropy per row ----------------------------------
    def xent(y_onehot, p):
        p = np.clip(p, eps, 1 - eps)
        return -np.nansum(y_onehot * np.log(p), axis=1) # (n,)
    
    residual_loo = xent(Yk, mu_loo)
    residual_loco1 = xent(Yk, mu_loco1)
    residual_loco2 = xent(Yk, mu_loco2)
    residual_loco12 = xent(Yk, mu_loco12)
    
    return residual_loco12, residual_loco1, residual_loco2, residual_loo



def getCI(delta_cap, alpha=0.1):
    sigma = np.std(delta_cap, ddof=1)
    ci = norm.ppf(1 - alpha / 2) * sigma / np.sqrt(len(delta_cap))
    return ci


def featureInteractions(X, Y, n_ratio, m_ratio, B, model, 
                        predictions, mp_observations, mp_features,
                        feature_pairs, alpha=0.1, bonferroni=False):
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
    #predictions, mp_observations, mp_features = predict(X, Y, n_ratio, m_ratio, B, model)
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
    r12, r1, r2, r = computeDeltaCap_xent(Y, j1, j2, predictions, mp_observations, mp_features)
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

def featureInteractions_parallel(X, Y, n_ratio, m_ratio, B, 
                        predictions, mp_observations, mp_features, 
                        model, feature_pairs, alpha=0.1, bonferroni=False, n_jobs=None):
    """
    Parallelized computation of interaction metrics (iLOCO) for feature pairs.

    Parameters:
    - n_jobs: number of processes (default: all cores)

    Returns:
    - dict: results for each (j1, j2) pair
    """
    #predictions, mp_observations, mp_features = predict(X, Y, n_ratio, m_ratio, B, model)

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
