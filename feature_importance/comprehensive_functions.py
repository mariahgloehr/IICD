import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import clone
import itertools

##---------
## mp ensemble functions
##---------
def buildMP(X, Y, n_ratio, m_ratio, adjust_col=None):
    """
    Builds a minipatch. 
    Parameters:
    X (pandas.DataFrame): Feature matrix.
    Y (pandas.DataFrame): Target variable.
    n_ratio (float): Ratio parameter for predict function.
    m_ratio (float): Ratio parameter for predict function.
    adjust_col (int): If provided, adjust for feature by forcing it into every minipatch
    
    Returns:
    - idx_I = the chosen subset of observation indices
    - idx_F = the chosen subset of feature indices (always includes adjust_col if given)
    - x_mp  = the minipatch of observations
    - y_mp  = the minipatch of responses
    """

    N, M = len(X), len(X[0])
    n = int(n_ratio * N)
    m = int(m_ratio * M)
    
    # uniformly sample a subset of observations
    idx_I = np.random.choice(N, size=n, replace=False)
    idx_I.sort()

    if adjust_col is not None:
        # sample remaining features excluding the adjust column
        candidate_features = [j for j in range(M) if j != adjust_col]
        sampled = np.random.choice(candidate_features, size=m-1, replace=False)
        idx_F = np.sort(np.append(sampled, adjust_col))
    else:
        idx_F = np.random.choice(M, size=m, replace=False)
        idx_F.sort()

    ## record which obs/features are subsampled 
    x_mp = X[np.ix_(idx_I, idx_F)]
    y_mp = Y[np.ix_(idx_I)]
    return idx_I, idx_F, x_mp, y_mp

def mp_ensemble(X, Y, n_ratio, m_ratio, B, models, adjust_col=None):

    """
    Builds, fits, and predicts a minipatch ensemble

    Parameters:
    X (pandas.DataFrame): Feature matrix.
    Y (pandas.DataFrame): Target variable.
    n_ratio (float): Ratio parameter for predict function.
    m_ratio (float): Ratio parameter for predict function.
    B (int): Number of bootstrap samples.
    models (list): List of unfitted scikit-learn model objects
    adjust_col (int): If provided, adjust for feature by forcing it into every minipatch

    Returns:
    - predictions: (N, B) array of predictions from each model
    - mp_observations: (N, B) boolean array marking sampled observations
    - mp_features: (M, B) boolean array marking sampled features
    """

    # Make X, Y numpy arrays
    X = X.to_numpy()
    Y = Y.to_numpy().ravel()

    N, M = X.shape
    mp_observations = np.zeros((N, B), dtype=bool)
    mp_features = np.zeros((M, B), dtype=bool)
    predictions = np.empty((N, B))
    ensemble = [None] * B

    for b in range(B):
        # build one minipatch
        idx_I, idx_F, x_mp, y_mp = buildMP(X, Y, n_ratio, m_ratio, adjust_col)

        # randomly select and a model
        base_model = np.random.choice(models)
        model = clone(base_model).fit(x_mp, y_mp)

        # store fitted model
        ensemble[b] = model

        # record which samples/features were used
        mp_observations[idx_I, b] = True
        mp_features[idx_F, b] = True  

        # Predict on full X using the features sampled in this minipatch
        predictions[:, b] = model.predict(X[:, idx_F])

    return {
        "predictions": predictions,
        "mp_observations": mp_observations,
        "mp_features": mp_features,
        "ensemble": ensemble
    }

##-----
## feature groups function
##-----

def make_feature_groups(X, order=2, include_col=None, subset=None):
    """
    Generate feature index pairs or triplets for interaction analysis.

    Parameters:
    X (pandas.DataFrame): Feature matrix.
    order(int): Order of interactions to generate (2 = pairs, 3 = triplets)
    include_col (int or None): If provided, only include groups that contain this feature index
    subset(list[int] or None): If provided, limit group generation to only these feature indices

    Returns:
    list[tuple[int]]
        List of feature index pairs or triplets.
    """
    # Total number of features (columns in X as numpy array)
    X = X.to_numpy()
    M = X.shape[1]

    # Define feature pool
    if subset is not None:
        features = subset
    else:
        features = list(range(M))

    # Generate all possible combinations of the chosen order
    groups = list(itertools.combinations(features, order))

    # Filter based on include_col logic
    if include_col is not None:
        groups = [g for g in groups if include_col in g]

    return groups

##-----
## feature interaction functions
##-----

def _to_one_hot(y, K=None):
    """Coerce labels to one-hot. Works with pandas Categorical, strings, ints."""
    y = np.asarray(y)
    if y.ndim == 2: # already one-hot
        K = y.shape[1]
        return y.astype(float), K
    
    # If y is non-numeric, map to integers 0..K-1
    if not np.issubdtype(y.dtype, np.number):
        # unique stable mapping
        uniques, inv = np.unique(y, return_inverse=True)
        y = inv

    if K is None:
        K = int(np.nanmax(y)) + 1
    
    Yk = np.zeros((y.shape[0], K), dtype=float)
    Yk[np.arange(y.shape[0]), y.astype(int)] = 1.0
    return Yk, K

def _as_probabilities(P, K_hint=None):
    """
    Normalize per-patch predictions into probabilities with shape:
    - binary: (n,B,2)
    - multiclass: (n,B,K)
    Accepts:
    - (n,B,K) probs (we re-normalize on last axis)
    - (n,B) in {0,1} hard labels -> one-hot then average later
    - (n,B) floats in [0,1] -> binary probs (p1); expand to (p0,p1)
    - (n,B) real-valued logits -> apply sigmoid -> probs
    """
    P = np.asarray(P)
    if P.ndim == 3:
        n,B,K = P.shape
        # re-normalize defensively
        denom = np.sum(P, axis=-1, keepdims=True)
        denom[denom == 0] = np.nan
        return P / denom, K
    
    if P.ndim != 2:
        raise ValueError("`predictions` must be 2D (n,B) or 3D (n,B,K).")
    
    # 2D case
    n,B = P.shape
    # detect int-like (hard labels)
    is_int_like = np.all(np.equal(P, np.round(P)))
    if is_int_like and np.isfinite(P).all() and P.min() >= 0:
        max_label = int(P.max())
        K = max(2, max_label + 1)
        preds_k = np.zeros((n,B,K), dtype=float)
        preds_k[np.arange(n)[:,None], np.arange(B)[None,:], P.astype(int)] = 1.0
        return preds_k, K

    # not int-like: try to detect binary probabilities vs logits
    if np.nanmin(P) >= 0.0 and np.nanmax(P) <= 1.0:
        # treat as binary probabilities p1
        p1 = P
        p0 = 1.0 - p1
        return np.stack([p0, p1], axis=-1), 2
    
    # looks like logits -> sigmoid to binary probs
    p1 = 1.0 / (1.0 + np.exp(-P))
    p0 = 1.0 - p1
    return np.stack([p0, p1], axis=-1), 2

def computeDeltaCap_xent_second(Y, j1, j2, predictions, mp_observations, mp_features, eps: float = 1e-12
):
    """
    Cross-entropy LO(O/CO) for classification.
    Handles categorical/string Y and mixed prediction formats.
    """
    # 1) Normalize labels to one-hot first (avoids `1 - Y` entirely)
    Yk, K_y = _to_one_hot(Y)

    # 2) Normalize predictions to probs with explicit K
    preds_k, K_p = _as_probabilities(predictions, K_hint=Yk.shape[1])

    # 3) Ensure class dimension matches
    if K_p != K_y:
        # If preds have 2 classes but Y has >2, or vice versa, this is a true mismatch
        raise ValueError(f"Class count mismatch between labels (K={K_y}) and predictions (K={K_p}).")
    
    n, B = mp_observations.shape
    assert preds_k.shape[0] == n and preds_k.shape[1] == B

    def masked_mean_probs(preds, mask_bool):
        w = mask_bool.astype(float) # (n,B)
        denom = w.sum(axis=1, keepdims=True) # (n,1)
        denom[denom == 0] = np.nan
        num = np.nansum(preds * w[..., None], axis=1) # (n,K)
        return num / denom
    
    # Masks (same as your regression version)
    loo = 1 - mp_observations
    mu_loo = masked_mean_probs(preds_k, loo)

    loco1_mask = loo * (1 - mp_features[j1, :])
    mu_loco1 = masked_mean_probs(preds_k, loco1_mask)

    loco2_mask = loo * (1 - mp_features[j2, :])
    mu_loco2 = masked_mean_probs(preds_k, loco2_mask)

    loco12_mask = loco1_mask * loco2_mask
    mu_loco12 = masked_mean_probs(preds_k, loco12_mask)

    def xent(y_onehot, p):
        p = np.clip(p, eps, 1 - eps)
        return -np.nansum(y_onehot * np.log(p), axis=1)
    
    residual_loo = xent(Yk, mu_loo)
    residual_loco1 = xent(Yk, mu_loco1)
    residual_loco2 = xent(Yk, mu_loco2)
    residual_loco12 = xent(Yk, mu_loco12)
    return residual_loco12, residual_loco1, residual_loco2, residual_loo


def computeDeltaCap_second(Y, j1, j2, predictions, mp_observations, mp_features, 
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


def computeDeltaCap_xent_third(Y, j1, j2, j3, predictions, mp_observations, mp_features, eps: float = 1e-12):
    """
    Third-order cross-entropy DeltaCap.
    """
    # Normalize labels and predictions
    Yk, K_y = _to_one_hot(Y)
    preds_k, K_p = _as_probabilities(predictions, K_hint=Yk.shape[1])
    if K_p != K_y:
        raise ValueError(f"Class count mismatch between labels (K={K_y}) and predictions (K={K_p}).")
    n, B = mp_observations.shape

    def masked_mean_probs(preds, mask_bool):
        w = mask_bool.astype(float)
        denom = w.sum(axis=1, keepdims=True)
        denom[denom == 0] = np.nan
        num = np.nansum(preds * w[..., None], axis=1)
        return num / denom

    loo = 1 - mp_observations
    m1 = 1 - mp_features[j1, :]
    m2 = 1 - mp_features[j2, :]
    m3 = 1 - mp_features[j3, :]

    # all leave-out combinations
    mu_loo     = masked_mean_probs(preds_k, loo)
    mu_loco1   = masked_mean_probs(preds_k, loo * m1)
    mu_loco2   = masked_mean_probs(preds_k, loo * m2)
    mu_loco3   = masked_mean_probs(preds_k, loo * m3)
    mu_loco12  = masked_mean_probs(preds_k, loo * m1 * m2)
    mu_loco13  = masked_mean_probs(preds_k, loo * m1 * m3)
    mu_loco23  = masked_mean_probs(preds_k, loo * m2 * m3)
    mu_loco123 = masked_mean_probs(preds_k, loo * m1 * m2 * m3)

    def xent(y_onehot, p):
        p = np.clip(p, eps, 1 - eps)
        return -np.nansum(y_onehot * np.log(p), axis=1)

    # compute residuals for each subset
    r_loo     = xent(Yk, mu_loo)
    r_loco1   = xent(Yk, mu_loco1)
    r_loco2   = xent(Yk, mu_loco2)
    r_loco3   = xent(Yk, mu_loco3)
    r_loco12  = xent(Yk, mu_loco12)
    r_loco13  = xent(Yk, mu_loco13)
    r_loco23  = xent(Yk, mu_loco23)
    r_loco123 = xent(Yk, mu_loco123)

    return (r_loco123, r_loco12, r_loco13, r_loco23,
            r_loco1, r_loco2, r_loco3, r_loo)


def computeDeltaCap_third(Y, j1, j2, j3, predictions, mp_observations, mp_features, metric=np.square):
    """
    Computes the squared error vectors from LOCO and LOO predictions for three variables.
    """
    # Leave-One-Out (LOO) predictions
    loo = 1 - mp_observations
    mu_loo = np.sum(predictions * loo, axis=1) / np.sum(loo, axis=1)

    # Leave-One-Covariate-Out (LOCO) predictions for each variable and combinations
    loco1 = loo * (1 - mp_features[j1, :])
    mu_loco1 = np.sum(predictions * loco1, axis=1) / np.sum(loco1, axis=1)

    loco2 = loo * (1 - mp_features[j2, :])
    mu_loco2 = np.sum(predictions * loco2, axis=1) / np.sum(loco2, axis=1)

    loco3 = loo * (1 - mp_features[j3, :])
    mu_loco3 = np.sum(predictions * loco3, axis=1) / np.sum(loco3, axis=1)

    loco12 = loco1 * loco2
    mu_loco12 = np.sum(predictions * loco12, axis=1) / np.sum(loco12, axis=1)

    loco13 = loco1 * loco3
    mu_loco13 = np.sum(predictions * loco13, axis=1) / np.sum(loco13, axis=1)

    loco23 = loco2 * loco3
    mu_loco23 = np.sum(predictions * loco23, axis=1) / np.sum(loco23, axis=1)

    loco123 = loco1 * loco2 * loco3
    mu_loco123 = np.sum(predictions * loco123, axis=1) / np.sum(loco123, axis=1)

    # Residuals
    residual_loo = metric(Y - mu_loo)
    residual_loco1 = metric(Y - mu_loco1)
    residual_loco2 = metric(Y - mu_loco2)
    residual_loco3 = metric(Y - mu_loco3)
    residual_loco12 = metric(Y - mu_loco12)
    residual_loco13 = metric(Y - mu_loco13)
    residual_loco23 = metric(Y - mu_loco23)
    residual_loco123 = metric(Y - mu_loco123)

    return (residual_loco123, residual_loco1, residual_loco2, residual_loco3,
            residual_loco12, residual_loco13, residual_loco23, residual_loo)

def getCI(delta_cap, alpha=0.1):
    """Get confidence interval width / 2
    """
    sigma = np.std(delta_cap, ddof=1)
    ci = norm.ppf(1 - alpha / 2) * sigma / np.sqrt(len(delta_cap))
    return ci


def featureInteractions(X, Y, mp_ensemble, feature_groups, 
                        order = 2, type = "regression", alpha = 0.1, bonferroni=False):
    """
    Computes interaction metrics (iLOCO) for multiple feature triplets.

    Parameters:
    X (pandas.DataFrame): Feature matrix.
    Y (pandas.DataFrame): Target variable.
    mp_ensemble: pre-made minipatch ensemble (with predictions) from the mp_ensemble() function
    feature_groups (list of tuples): List of tuples where each tuple contains either two (j1,j2)
     or three feature indices (j1, j2, j3).
    order (int): Integer for second or third order iLOCO.
    type (string): regression or classification
    alpha (int):
    bonferroni (boolean):

    Returns:
    dict: Dictionary where keys are feature pairs (j1, j2) or triplets (j1, j2, j3) and values are dictionaries containing:
          - 'iloco': The mean interaction value.
          - iloco_max, iloco_ratio, ci
    """
    # Store feature names from X as a DataFrame, then convert to numpy array
    feature_names = X.columns.tolist()
    X = X.to_numpy()
    
    Y = Y.to_numpy().ravel()

    # Get predictions and model properties from pre-made ensemble
    predictions = mp_ensemble["predictions"]
    mp_observations = mp_ensemble["mp_observations"]
    mp_features = mp_ensemble["mp_features"]
    
    # Initialize results dictionary to store interaction metrics for each feature triplet
    results = {}

    # Adjust alpha if Bonferroni is requested
    if bonferroni:
        adjusted_alpha = alpha / len(feature_groups)
    else:
        adjusted_alpha = alpha
        
    if order == 2:
        feat_key = ",".join(feature_names[j] for j in (j1, j2))
        if type == "regression":
            for (j1, j2) in feature_groups:
                # Compute DeltaCap for the current feature pair
                r12, r1, r2, r = computeDeltaCap_second(Y, j1, j2, predictions, mp_observations, mp_features)
                dc = r1 + r2 - r12 - r
                iloco = np.mean(dc)
                iloco_max = max(0, iloco)
                iloco_ratio = iloco / np.mean(r)
                dc = r1 - r
                ci = getCI(dc, adjusted_alpha)

                # Store results for the current feature pair in the dictionary
                results[(j1, j2)] = {
                'features': feat_key,
                'iloco': iloco,
                'iloco_max': iloco_max,
                'iloco_ratio': iloco_ratio,
                'ci': ci
                }

        if type == "classification":
            for (j1, j2) in feature_groups:
                # Compute DeltaCap_xent for the current feature pair
                r12, r1, r2, r = computeDeltaCap_xent_second(Y, j1, j2, predictions, mp_observations, mp_features)
                dc = r1 + r2 - r12 - r
                iloco = np.mean(dc)
                iloco_max = max(0, iloco)
                iloco_ratio = iloco / np.mean(r)
                dc = r1 - r
                ci = getCI(dc, adjusted_alpha)

                # Store results for the current feature pair in the dictionary
                results[(j1, j2)] = {
                    'features': feat_key,
                    'iloco': iloco,
                    'iloco_max': iloco_max,
                    'iloco_ratio': iloco_ratio,
                    'ci': ci
                }

    if order == 3:
        feat_key = ",".join(feature_names[j] for j in (j1, j2, j3))
        if type == "regression":
         # Loop over each feature triplet in the list using regression delta cap
            for (j1, j2, j3) in feature_groups:
                # Compute DeltaCap for the current feature triplet
                r123, r1, r2, r3, r12, r13, r23, r = computeDeltaCap_third(Y, j1, j2, j3, predictions, mp_observations, mp_features)
                dc = r1 + r2 + r3 - r12 - r13 - r23 + r123 - r
                iloco = np.mean(dc)
                iloco_max = max(0, iloco)
                iloco_ratio = iloco / np.mean(r)
                dc = r1 - r
                ci = getCI(dc, adjusted_alpha)

                # Store results for the current feature pair in the dictionary
                results[(j1, j2)] = {
                    'features': feat_key,
                    'iloco': iloco,
                    'iloco_max': iloco_max,
                    'iloco_ratio': iloco_ratio,
                    'ci': ci
                }

        if type == "classification":
        # Loop over each feature triplet in the list using  deltacap_xent
            for (j1, j2, j3) in feature_groups:
                # Compute DeltaCap for the current feature triplet
                r123, r1, r2, r3, r12, r13, r23, r = computeDeltaCap_xent_third(Y, j1, j2, j3, predictions, mp_observations, mp_features)
                dc = r1 + r2 + r3 - r12 - r13 - r23 + r123 - r
                iloco = np.mean(dc)
                iloco_max = max(0, iloco)
                iloco_ratio = iloco / np.mean(r)
                dc = r1 - r
                ci = getCI(dc, adjusted_alpha)

                # Store results for the current feature triplet in the dictionary
                results[(j1, j2, j3)] = {
                    'features': feat_key,
                    'iloco': iloco,
                    'iloco_max': iloco_max,
                    'iloco_ratio': iloco_ratio,
                    'ci': ci
                }


    return results