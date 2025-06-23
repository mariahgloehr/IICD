from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from scipy import stats
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV

# Define Baseline Regressor to be used later
def DecisionTreeReg(X,Y,X1):
    tree = DecisionTreeRegressor().fit(X,Y)
    return tree.predict(X1)
def ridgecv(X,Y,X1):
    clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1], fit_intercept = False).fit(X, Y)    
    return clf.predict(X1)

def ridge2(X,Y,X1):
    clf = Ridge(fit_intercept = False,alpha=0.001).fit(X, Y)    
    return clf.predict(X1)

def RFreg(X,Y,X1):
    rf = RandomForestRegressor(n_estimators=200,criterion='mse',bootstrap=False).fit(X,Y)
    return rf.predict(X1)

def kernelSVR(X,Y,X1):
    krr = SVR(kernel='rbf',C=1000).fit(X, Y)
    return krr.predict(X1)


##############################
##### Classification
##############################

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from  sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
def logitridge(X,Y):
    fit=LogisticRegression(penalty='l2',solver='saga',max_iter=10,C = 1000).fit(X,Y)
    return fit

def DecisionTreeClass(X,Y):
    tree = DecisionTreeClassifier(min_samples_leaf=5).fit(X,Y)
    return tree


def logitridgecv(X,Y):
    fit=LogisticRegressionCV(cv=5,penalty='l2',solver='saga').fit(X,Y)
    return fit

def RFclass(X,Y):
    M =len(X[0])    # when bootstrap=False, it means each tree is trained on all rows of X and only
    #      subsamples its columns (which are features).
    rf = RandomForestClassifier(n_estimators=200, bootstrap=False).fit(X,Y)
    return rf


def kernelSVC(X,Y,ridge_mult=0.001):
    krr = SVC(kernel='rbf',C=1/ridge_mult,probability=True).fit(X, Y)
    return krr







