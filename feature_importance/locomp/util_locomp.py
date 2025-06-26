import pandas as pd
import numpy as np
from scipy.stats import norm





def buildMPReg(X,Y,n_ratio,m_ratio):
    N = len(X)
    M = len(X[0])
    n = int(np.round(n_ratio * N))
    m = int(np.round(m_ratio * M))
    r = np.random.RandomState()
    ## index of minipatch
    idx_I = np.sort(r.choice(N, size=n, replace=False)) # uniform sampling of subset of observations
    idx_F = np.sort(r.choice(M, size=m, replace=False)) # uniform sampling of subset of features
    ## record which obs/features are subsampled 
    x_mp=X[np.ix_(idx_I, idx_F)]
    y_mp=Y[np.ix_(idx_I)]
    return [idx_I,idx_F,x_mp,y_mp]
def buildMPClass(X,Y,n_ratio,m_ratio):
    N = len(X)
    M = len(X[0])
    n = int(np.ceil(n_ratio * N))
    m = int(np.ceil(m_ratio * M))
    r = np.random.RandomState()
    ## index of minipatch
    #3 stratified sampling
    Y_pd=pd.DataFrame(Y.reshape((len(Y),1)))

    idx_I =Y_pd.groupby(0, group_keys=False).apply(lambda x: x.sample(frac=n_ratio))
    idx_I = np.sort(list(idx_I.index)) # stratified sampling of subset of observations
    idx_F = np.sort(r.choice(M, size=m, replace=False)) # uniform sampling of subset of features
    ## record which obs/features are subsampled 
    x_mp=X[np.ix_(idx_I, idx_F)]
    y_mp=Y[np.ix_(idx_I)]
    return [idx_I,idx_F,x_mp,y_mp]


def predictMPReg(X,Y,X1, n_ratio,m_ratio,B,fit_func):
    N = len(X)
    M = len(X[0])
    N1 = len(X1)
    in_mp_obs,in_mp_feature = np.zeros((B,N),dtype=bool),np.zeros((B,M),dtype=bool)
    predictions=[]
    for b in range(B):        
        [idx_I,idx_F,x_mp,y_mp] = buildMPReg(X,Y,n_ratio,m_ratio)
        predictions.append(fit_func(x_mp,y_mp,X1[:, idx_F]))
        in_mp_obs[b,idx_I]=True
        in_mp_feature[b,idx_F]=True  
    return [np.array(predictions),in_mp_obs,in_mp_feature]


def predictMPClass(X,Y,X1, n_ratio,m_ratio,B,fit_func):
    N = len(X)
    M = len(X[0])
    N1 = len(X1)
    clas=list(set(Y))

    in_mp_obs,in_mp_feature = np.zeros((B,N),dtype=bool),np.zeros((B,M),dtype=bool)
    predictions=[]
    for b in range(B):
        [idx_I,idx_F,x_mp,y_mp] = buildMPClass(X,Y,n_ratio,m_ratio)
        model = fit_func(x_mp,y_mp)
        prob = pd.DataFrame(model.predict_proba(X1[:, idx_F]), columns=list(set(y_mp)))
        for i in (clas):
            if i not in prob.columns:
                prob[i]=0
        #prob = prob[clas]
    ############################################
        predictions.append(np.array(prob))
        in_mp_obs[b,idx_I]=True
        in_mp_feature[b,idx_F]=True  
    return [np.array(predictions),in_mp_obs,in_mp_feature]

def getNC(true_y,prob,method = 'prob1'):
    if method=='prob2':
        if len(true_y)==1:
            true_y=true_y[0]
            py=prob[true_y]
            pz = max(prob.drop(true_y,axis=1))
            nc = (1- py+pz)/2
        else:
            py=[prob[item][i] for i,item in enumerate(true_y)] ##prob of true label
            pz=[max(prob.iloc[i].drop(true_y[i])) for i in range(len(true_y))] ## max prob of other label 
            nc = [(1- py[i]+pz[i])/2 for i in range(len(py))]
    if method=='prob1':
        if len(true_y)==1:
            true_y=int(true_y[0])
            py=prob[true_y]
            nc = (1- py)
        else:
            py=[prob[item][i] for i,item in enumerate(true_y)] ##prob of true label
            nc = [(1- py[i]) for i in range(len(py))]
    return np.array(nc)

def ztest(z,alpha,MM=1,bonf_correct=True):
    try:
        s = np.std(z)
    except:
        return [0,0,0,0]
    
    l = len(z)
    s = np.std(z)
    if s==0:
        return [0,0,0,0]
    m = np.mean(z)
    pval1 = 1-norm.cdf(m/s*np.sqrt(l))

    pval2 = 2*(1-norm.cdf(np.abs(m/s*np.sqrt(l))))

    # Apply Bonferroni correction for M tests
    if bonf_correct:
        pval1= min(MM*pval1,1)
        pval2= min(MM*pval2,1)
        alpha = alpha/MM
    q = norm.ppf(1-alpha/2)
    left  = m - q*s/np.sqrt(l)
    right = m + q*s/np.sqrt(l)
    return [pval1,pval2, left,right,m]

def ztest_adjust(z,alpha,var=0.00001,MM=1,bonf_correct=True):
    l = len(z)
    s = np.std(z)
    m = np.mean(z)
    pval1 = 1-norm.cdf(m/(s/np.sqrt(l)+var))

    pval2 = 2*(1-norm.cdf(np.abs(m/s/np.sqrt(l)+var)))

    # Apply Bonferroni correction for M tests
    if bonf_correct:
        pval1= min(MM*pval1,1)
        pval2= min(MM*pval2,1)
        alpha = alpha/MM
    q = norm.ppf(1-alpha/2)
    left  = m - q*(s/np.sqrt(l)+var)
    right = m + q*(s/np.sqrt(l)+var)
    return [pval1,pval2, left,right]
