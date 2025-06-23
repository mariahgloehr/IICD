import numpy as np
import pandas as pd
from scipy.stats import *
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy.random as r
from random import sample
from joblib import Parallel, delayed
from .util_locomp import buildMPClass,predictMPClass,ztest,getNC

class LOCOMPClass():
    """ 
    Parameters
    ----------
    data: 
        X: N*M numpy array.
    label:    
        Y: N*1 numpy array.
        
    fit_func : classifition estimator object
        This is assumed to take training data X,Y, and test data X1 as input, and will output minipatch prediction of X1. 
   
        
    selected_features: a list of values which indictates indices of features of interests.  
    
    
    alpha: float, user-defined error rate.
    
    
    bonf: str, indicator of bonferroni correction.
        
        
        
    Attributes
    ----------
    
    loco_ci: a list of LOCO featuera importance interval
    
    info: a list of LOCO featuera inference related values
    
    diff: a list of minipatch prediction stability measures
    
    """
    
    def __init__(self,X,Y,n_ratio,m_ratio,B,fit_func, selected_features=[0],alpha=0.1,bonf=True):
        self.X=X
        self.Y=Y
        self.n_ratio=n_ratio
        self.m_ratio=m_ratio
        self.B = B
        self.fit_func=fit_func
        self.selected_features=selected_features
        self.alpha=alpha
        self.bonf=bonf
        
    def run_loco(self,*args,**kwargs):
        N=len(self.X)
        M = len(self.X[0])
        clas=np.unique(self.Y)

        [predictions,in_mp_obs,in_mp_feature]= predictMPClass(self.X,self.Y,self.X,self.n_ratio,self.m_ratio,self.B,self.fit_func)
        zeros=False

        #############################
        ## Find LOO
        ############################
        diff=[]
        b_keep = pd.DataFrame(~in_mp_obs).apply(lambda i: np.array(i[i].index))

        #############################
        ## Find LOO
        ############################
        for i in range(N):
            #####################
            ###### estimate B
            sel_2 = np.array(sample(list(b_keep[i]),20))
            sel_2.shape = (2,10)
            diff.append(np.square(predictions[sel_2[0],i][:,0] - predictions[sel_2[1],i][:,0]).mean())

        with_j = map(lambda i: predictions[b_keep[i],i].mean(0),range(N))
        with_j = pd.DataFrame(list(with_j), columns=clas)
        resids_LOO = getNC(self.Y, with_j)

        ################################
        ######## FIND LOCO
        #############################
        def get_loco(i,j):
            b_keep_f = list(set(np.argwhere(~(in_mp_feature[:,j])).reshape(-1)) & set(np.argwhere(~(in_mp_obs[:,i])).reshape(-1)))
            return predictions[b_keep_f,i].mean(0)
        
        if len(self.selected_features)==0:
            ff = list(range(M))
        else:
            ff=self.selected_features
        results = Parallel(n_jobs=-1)(delayed(get_loco)(i,j) for i in range(N) for j in range(M))
        ress = pd.DataFrame(results)
        ress['i'] = np.repeat(range(N),M)
        ress['j'] = np.tile(range(M),N)
        ress['true_y'] = np.repeat(self.Y,M)
        ress['resid_loco'] = getNC(ress['true_y'], ress[[0,1]])
        ress['resid_loo'] = np.repeat(resids_LOO,M)
        ress['zz'] = ress['resid_loco'] -ress['resid_loo']


        inf_z = np.zeros((len(ff),4))
        for idd,j in enumerate(ff): 
            inf_z[idd] = ztest(ress[ress.j==idd].zz,self.alpha,MM=len(ff),bonf_correct =self.bonf)
        ###########################
        self.loco_ci=inf_z
        self.info=ress
        self.diff=diff

    def correct_variance(self,eps,*args,**kwargs):
        var = np.sqrt(np.mean(self.diff))*np.log(len(self.X))*self.n_ratio*eps
        return ztest_adjust(self.info['zz'],self.alpha, var = var)
            

