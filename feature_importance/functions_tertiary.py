import numpy as np
from scipy.stats import norm
from sklearn.base import clone
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split


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


def buildMP(X, Y, n_ratio, m_ratio, perturbation_col=None):
    """
    Builds a minipatch. Returns:
    - idx_I = the chosen subset of observation indices
    - idx_F = the chosen subset of feature indices (always includes perturbation_col if given)
    - x_mp  = the minipatch of observations
    - y_mp  = the minipatch of responses
    """

    N, M = len(X), len(X[0])
    n = int(n_ratio * N)
    m = int(m_ratio * M)
    
    # uniformly sample a subset of observations
    idx_I = np.random.choice(N, size=n, replace=False)
    idx_I.sort()

    if perturbation_col is not None:
        # sample remaining features excluding the perturbation column
        candidate_features = [j for j in range(M) if j != perturbation_col]
        sampled = np.random.choice(candidate_features, size=m-1, replace=False)
        idx_F = np.sort(np.append(sampled, perturbation_col))
    else:
        idx_F = np.random.choice(M, size=m, replace=False)
        idx_F.sort()

    ## record which obs/features are subsampled 
    x_mp = X[np.ix_(idx_I, idx_F)]
    y_mp = Y[np.ix_(idx_I)]
    return idx_I, idx_F, x_mp, y_mp

class Ensemble:
    def __init__(self, models):
        self.models = models

    def fit(self, X, Y, n_ratio, m_ratio, B, perturbation_col=None):
        N, M = X.shape
        self.mp_observations = np.zeros((N, B), dtype=bool)
        self.mp_features = np.zeros((M, B), dtype=bool)
        self.ensemble = [None] * B
        for b in range(B):
            idx_I, idx_F, x_mp, y_mp = buildMP(X, Y, n_ratio, m_ratio, perturbation_col)
            # randomly select a model each time
            model = np.random.choice(self.models)
            
            self.ensemble[b] = clone(model).fit(x_mp, y_mp) 
            self.mp_observations[idx_I, b] = True
            self.mp_features[idx_F, b] = True  
        return self
        
    def predict(self, X):
        predictions = np.empty((len(X), len(self.ensemble)))
        for b, m in enumerate(self.ensemble):
            predictions[:, b] = m.predict(X[:, self.mp_features[:, b]])
        return predictions
    


def predict(X, Y, n_ratio, m_ratio, B, models, perturbation_col):
    """
    Fits and predicts models
    """
    N, M = X.shape
    mp_observations = np.zeros((N, B), dtype=bool)
    mp_features = np.zeros((M, B), dtype=bool)
    predictions = np.empty((N, B))
    for b in range(B):
        idx_I, idx_F, x_mp, y_mp = buildMP(X, Y, n_ratio, m_ratio, perturbation_col)

        base_model = np.random.choice(models)
        model = clone(base_model).fit(x_mp, y_mp)

        predictions[:, b] = model.fit(x_mp, y_mp).predict(X[:, idx_F])
        mp_observations[idx_I, b] = True
        mp_features[idx_F, b] = True  
    return predictions, mp_observations, mp_features


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



def getCI(delta_cap, alpha=0.1):
    """Get confidence interval width / 2
    """
    sigma = np.std(delta_cap, ddof=1)
    ci = norm.ppf(1 - alpha / 2) * sigma / np.sqrt(len(delta_cap))
    return ci


def computeDeltaCap(Y, j1, j2, j3, predictions, mp_observations, mp_features, metric=np.square):
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


def featureInteractions(X, Y, n_ratio, m_ratio, B, models, feature_triplets, 
                        perturbation_col = None, alpha = 0.1, bonferroni=False):
    """
    Computes interaction metrics (iLOCO) for multiple feature triplets.

    Parameters:
    X (numpy.ndarray): Feature matrix.
    Y (numpy.ndarray): Target variable.
    n_ratio (float): Ratio parameter for predict function.
    m_ratio (float): Ratio parameter for predict function.
    B (int): Number of bootstrap samples.
    model (object): Trained model to make predictions.
    feature_triplets (list of tuples): List of tuples where each tuple contains three feature indices (j1, j2, j3).

    Returns:
    dict: Dictionary where keys are feature triplets (j1, j2, j3) and values are dictionaries containing:
          - 'iloco': The mean interaction value.
    """
    # Get predictions and model properties
    predictions, mp_observations, mp_features = predict(X, Y, n_ratio, m_ratio, B, models, perturbation_col)
    
    # Initialize results dictionary to store interaction metrics for each feature triplet
    results = {}

    # Adjust alpha if Bonferroni is requested
    if bonferroni:
        adjusted_alpha = alpha / len(feature_triplets)
    else:
        adjusted_alpha = alpha

    # Loop over each feature triplet in the list
    for (j1, j2, j3) in feature_triplets:
        # Compute DeltaCap for the current feature triplet
        r123, r1, r2, r3, r12, r13, r23, r = computeDeltaCap(Y, j1, j2, j3, predictions, mp_observations, mp_features)
        dc = r1 + r2 + r3 - r12 - r13 - r23 + r123 - r
        iloco = np.mean(dc)
        iloco_max = max(0, iloco)
        iloco_ratio = iloco / np.mean(r)
        dc = r1 - r
        ci = getCI(dc, adjusted_alpha)

        # Store results for the current feature triplet in the dictionary
        results[(j1, j2, j3)] = {
            'iloco': iloco,
            'iloco_max': iloco_max,
            'iloco_ratio': iloco_ratio,
            'ci': ci
        }

    return results








