###This script was checked on Mar 2023

###This script creates csvs files of the full G MATRIX 
# as well as the Pleiotropy portion and the LD portion
# for the sexual simulations 211, 212

from cmath import log
import os
from pprint import pprint
from statistics import variance
import sys
from typing import List
import numpy as np
import pandas as pd
import scipy
import scipy.stats as stats
import random

verbose = True if len( sys.argv ) > 1 else False
print("Verbose is {}".format(verbose))

def hl(_:str):
    return "\033[95m{}\033[0m".format(_)

def hl1(_:str):
    return "\033[93m{}\033[0m".format(_)

def getrep(path: str, repnum: int = -1) -> pd.DataFrame:
    """Uploaded a replicate to memory"""
    reppath   = os.path.join(path, 'replicate{}.csv'.format(repnum))
    Replicate = pd.read_csv(reppath, index_col=False)
    print("Opened a {} replicate at {}.".format(np.array(Replicate).shape, reppath))
    # print(Replicate)
    return Replicate

def shuffle_weights_t(w_t: np.ndarray) -> np.ndarray:
    """w_t is  Z x N """
    w_t_P = np.copy(w_t)
    for z in range(w_t.shape[1]):
        np.random.shuffle(w_t_P[:, z])
    return w_t_P

def shuffle_contribs_t(contrib_t: np.ndarray) -> np.ndarray:
    """contrib_t is Z x M x N"""
    contrib_t_P   = np.copy(contrib_t)
    [Z, M, N]     = contrib_t.shape
    _             = []
    shuffled_MNxZ = []

    for i in range(M):
        for j in range(N):
            c_ij = contrib_t_P[:, i, j]
            np.random.shuffle(c_ij)
            shuffled_MNxZ.append(c_ij)

    for i in range(Z):
        _.append(np.array(shuffled_MNxZ)[:, i].reshape((M, N)))

    return np.array(_)

def t_W_getall(df: pd.Series) -> np.ndarray:
    """get all weights from a timestamp as 1000x4"""
    return np.array(list(map(lambda e: np.fromstring(str(e), sep=',', dtype=float), df.iloc[2:1002])))

def t_C_getall(df: pd.Series) -> np.ndarray:
    """get all contributions from a timestamp as 1000x4x4"""
    return np.array(list(map(lambda _: np.fromstring(str(_), sep=',', dtype=float).reshape((4, 4)), df.iloc[1002:])))

def rep_covs(cor_exp_path, repn: int) -> List:
    """get average(over timestamps ) covariance matrices for replicate: [original, shuffled]"""

    r = getrep(cor_exp_path, repnum=repn)

    cov_all   = []
    cov_all_P = []

    for i, Timestamp in enumerate(r.iterrows()):
        # print(" \n------------------| Timestamp {} |---------------------".format(i))

        contribs_t = t_C_getall(Timestamp[1])
        weights_t = t_W_getall(Timestamp[1])
        # print(" --| Got Contribs & Weights  |-- ".format(i))
        # print("contributions :", np.shape(contribs_t))
        # print("weights:"       , np.shape(weights_t ))

        weights_t_P = shuffle_weights_t(weights_t)
        contribs_t_P = shuffle_contribs_t(contribs_t)
        # print(" --| Shuffled Contribs & Weights  {} |-- ".format(i))
        # print("shuffled contributions :", np.shape(contribs_t))
        # print("shuffled weights:"       , np.shape(weights_t ))

        phens_t = np.array([np.array(_[0]@_[1].T)
                           for _ in zip(contribs_t, weights_t)])
        phens_t_P = np.array([np.array(_[0]@_[1].T)
                             for _ in zip(contribs_t_P, weights_t_P)])
        # print(" --| Calculated Phenotypes for Timestamp {} |-- ".format(i))

        cov_all.append(np.cov(phens_t.T))        # originally, this has a .T
        cov_all_P.append(np.cov(phens_t_P.T))      # originally, this has a .T

    return [
        np.array(cov_all).mean(axis=0),
        np.array(cov_all_P).mean(axis=0)
    ]


do_N_iter            = 511



EXPERIMENT_DIR_COV   = '/scratch/st-whitlock-1/isa/csvs2.1.1' 
EXPERIMENT_DIR_UNCOV = '/scratch/st-whitlock-1/isa/csvs2.1.2' 

#? Correlated
exp_cor_covs     = np.array([[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]])
exp_cor_covs_P   = np.array([[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]])

#? Uncorrelated
exp_uncor_covs   = np.array([[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]])
exp_uncor_covs_P = np.array([[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]])

for _h in range(1, do_N_iter):
    print(
        "|--------------[Correlated] Replicate #{}---------------|".format(_h))
    try:

        u_cov_cor, u_cov_cor_P = rep_covs(os.path.join(EXPERIMENT_DIR_COV), _h)
        print(u_cov_cor.shape)
        exp_cor_covs= np.array([*exp_cor_covs,u_cov_cor])
        exp_cor_covs_P = np.array([*exp_cor_covs_P,u_cov_cor_P])
        
    except Exception as e:
        print("Skipped replicate {}".format(_h))
        print(e)
        ...
    print(
        "|--------------[Uncorrelated] Replicate #{}---------------|".format(_h))
    try:

        u_cov_uncor, u_cov_uncor_P = rep_covs(os.path.join(EXPERIMENT_DIR_UNCOV), _h)
        print(u_cov_uncor.shape)
        exp_uncor_covs= np.array([*exp_uncor_covs,u_cov_uncor])
        exp_uncor_covs_P = np.array([*exp_uncor_covs_P,u_cov_uncor_P])
    except Exception as e:
        print("Skipped replicate {}".format(_h))
        print(e)
        ...

exp_cor_covs = np.delete(exp_cor_covs, obj=[0,1], axis=0)
exp_cor_covs_P = np.delete(exp_cor_covs_P, obj=[0,1], axis=0)

exp_uncor_covs = np.delete(exp_uncor_covs, obj=[0,1], axis=0)
exp_uncor_covs_P = np.delete(exp_uncor_covs_P, obj=[0,1], axis=0)
#print(exp_uncor_covs.shape)
#print(exp_uncor_covs_P.shape)
####DELTA is the same as LD


lower_len = len(exp_cor_covs) if len(exp_cor_covs) < len(exp_uncor_covs) else len(exp_uncor_covs)

exp_cor_covs     = exp_cor_covs[:lower_len]
exp_uncor_covs   = exp_uncor_covs[:lower_len]
exp_cor_covs_P   = exp_cor_covs_P[:lower_len]
exp_uncor_covs_P = exp_uncor_covs_P[:lower_len]

####DELTA is the same as LD
delta_cor   = np.array(exp_cor_covs  ) - np.array(exp_cor_covs_P  )
delta_uncor = np.array(exp_uncor_covs) - np.array(exp_uncor_covs_P)
delta_cor   = np.array(exp_cor_covs  ) - np.array(exp_cor_covs_P  )
delta_uncor = np.array(exp_uncor_covs) - np.array(exp_uncor_covs_P)

COR_G_T = [exp_cor_covs[:,i,j] for i in range(0,4) for j in range(0,4)]
COR_G11_T, COR_G12_T, COR_G13_T, COR_G14_T, COR_G21_T, COR_G22_T, COR_G23_T, COR_G24_T, COR_G31_T, COR_G32_T, COR_G33_T, COR_G34_T, COR_G41_T, COR_G42_T, COR_G43_T, COR_G44_T = COR_G_T

COR_G_P = [exp_cor_covs_P[:,i,j] for i in range(0,4) for j in range(0,4)]
COR_G11_P, COR_G12_P, COR_G13_P, COR_G14_P, COR_G21_P, COR_G22_P, COR_G23_P, COR_G24_P, COR_G31_P, COR_G32_P, COR_G33_P, COR_G34_P, COR_G41_P, COR_G42_P, COR_G43_P, COR_G44_P = COR_G_P

COR_G_LD = [delta_cor[:,i,j] for i in range(0,4) for j in range(0,4)]
COR_G11_LD, COR_G12_LD, COR_G13_LD, COR_G14_LD, COR_G21_LD, COR_G22_LD, COR_G23_LD, COR_G24_LD, COR_G31_LD, COR_G32_LD, COR_G33_LD, COR_G34_LD, COR_G41_LD, COR_G42_LD, COR_G43_LD, COR_G44_LD = COR_G_LD


UNCOR_G_T = [exp_uncor_covs[:,i,j] for i in range(0,4) for j in range(0,4)]
UNCOR_G11_T, UNCOR_G12_T, UNCOR_G13_T, UNCOR_G14_T, UNCOR_G21_T, UNCOR_G22_T, UNCOR_G23_T, UNCOR_G24_T, UNCOR_G31_T, UNCOR_G32_T, UNCOR_G33_T, UNCOR_G34_T, UNCOR_G41_T, UNCOR_G42_T, UNCOR_G43_T, UNCOR_G44_T = UNCOR_G_T

UNCOR_G_P = [exp_uncor_covs_P[:,i,j] for i in range(0,4) for j in range(0,4)]
UNCOR_G11_P, UNCOR_G12_P, UNCOR_G13_P, UNCOR_G14_P, UNCOR_G21_P, UNCOR_G22_P, UNCOR_G23_P, UNCOR_G24_P, UNCOR_G31_P, UNCOR_G32_P, UNCOR_G33_P, UNCOR_G34_P, UNCOR_G41_P, UNCOR_G42_P, UNCOR_G43_P, UNCOR_G44_P = UNCOR_G_P

UNCOR_G_LD = [delta_uncor[:,i,j] for i in range(0,4) for j in range(0,4)]
UNCOR_G11_LD, UNCOR_G12_LD, UNCOR_G13_LD, UNCOR_G14_LD, UNCOR_G21_LD, UNCOR_G22_LD, UNCOR_G23_LD, UNCOR_G24_LD, UNCOR_G31_LD, UNCOR_G32_LD, UNCOR_G33_LD, UNCOR_G34_LD, UNCOR_G41_LD, UNCOR_G42_LD, UNCOR_G43_LD, UNCOR_G44_LD = UNCOR_G_LD

def GMAT_DFer(env,pre_DF, COVtype):
  """
  This is function that takes the type of data, for example "T", "P" or "LD" and the G matrix.
  It returns a dataframe
  """
### CORRELATED
  for i in range(1,5):
    for k in range (1,5):
        dict_key= "G{}{}".format(i,k)
        dict_val = eval("{}_G{}{}_{}".format(env,i,k,COVtype))
        pre_DF[dict_key] = dict_val
       
  return pre_DF

#Correlated
pre_DF_T = {'Environment': np.full(len(COR_G11_T), "SEX_Correlated_OTAL")}      
Correlated_111_T = pd.DataFrame(GMAT_DFer('COR',pre_DF_T,'T'))

pre_DF_P = {'Environment': np.full(len(COR_G11_P), "SEX_Correlated_PLEIOTROPY")}      
Correlated_111_P = pd.DataFrame(GMAT_DFer('COR',pre_DF_P,'P'))

pre_DF_LD = {'Environment': np.full(len(COR_G11_LD), "SEX_Correlated_LD")}      
Correlated_111_LD = pd.DataFrame(GMAT_DFer('COR',pre_DF_LD,'LD'))

Correlated_111_T.to_csv('SEX_Correlated_211_T.csv')
Correlated_111_P.to_csv('SEX_Correlated_211_P.csv')
Correlated_111_LD.to_csv('SEX_Correlated_211_LD.csv')

#UNCORRELATED
Upre_DF_T = {'Environment': np.full(len(UNCOR_G11_T), "SEX_Uncorrelated_TOTAL")}
Uncorrelated_112_T = pd.DataFrame(GMAT_DFer('UNCOR',Upre_DF_T,'T'))

Upre_DF_P = {'Environment': np.full(len(UNCOR_G11_P), "SEX_Uncorrelated_PLEIOTROPY")}
Uncorrelated_112_P = pd.DataFrame(GMAT_DFer('UNCOR',Upre_DF_P,'P'))

Upre_DF_LD = {'Environment': np.full(len(UNCOR_G11_LD), "SEX_Uncorrelated_LD")}
Uncorrelated_112_LD = pd.DataFrame(GMAT_DFer('UNCOR',Upre_DF_LD,'LD'))

Uncorrelated_112_T.to_csv('SEX_Uncorrelated_212_T.csv')
Uncorrelated_112_P.to_csv('SEX_Uncorrelated_212_P.csv')
Uncorrelated_112_LD.to_csv('SEX_Uncorrelated_212_LD.csv')

var_CORR_T = np.empty(0)
var_CORR_P = np.empty(0)
var_CORR_LD = np.empty(0)
var_UNCORR_T = np.empty(0)
var_UNCORR_P = np.empty(0)
var_UNCORR_LD = np.empty(0)


covar_CORR_T = np.empty(0)
covar_CORR_P = np.empty(0)
covar_CORR_LD = np.empty(0)
covar_UNCORR_T = np.empty(0)
covar_UNCORR_P = np.empty(0)
covar_UNCORR_LD = np.empty(0)


for i in range(1,5):
    var_CORR_T = np.append(var_CORR_T, globals("COR_G" + str(i) + str(i)+ "_T"))
    var_CORR_P = np.append(var_CORR_P, globals("COR_G" + str(i) + str(i)+ "_P"))
    var_CORR_LD = np.append(var_CORR_LD, globals("COR_G" + str(i) + str(i)+ "_LD"))
    var_UNCORR_T = np.append(var_UNCORR_T, globals("UNCOR_G" + str(i) + str(i)+ "_T"))
    var_UNCORR_P = np.append(var_UNCORR_P, globals("UNCOR_G" + str(i) + str(i)+ "_P"))
    var_UNCORR_LD = np.append(var_UNCORR_LD, eval("UNCOR_G" + str(i) + str(i)+ "_LD"))
    

for k in range(1,5):
    for j in range(1,5):
        if k == j:
            continue
        else:
            covar_UNCORR_T = np.append(covar_UNCORR_T, eval("UNCOR_G" + str(k) + str(j) + "_T"))
            covar_UNCORR_P = np.append(covar_UNCORR_P, eval("UNCOR_G" + str(k) + str(j) + "_P"))
            covar_UNCORR_LD = np.append(covar_UNCORR_LD, eval("UNCOR_G" + str(k) + str(j) + "_LD"))
            covar_CORR_T = np.append(covar_CORR_T, eval("COR_G" + str(k) + str(j) + "_T"))
            covar_CORR_P = np.append(covar_CORR_P, eval("COR_G" + str(k) + str(j) + "_P"))
            covar_CORR_LD = np.append(covar_CORR_LD, eval("COR_G" + str(k) + str(j) + "_LD"))
            
# #~~~~~~~~~Confidence Interval~~~~~~~~~~~~~~~~~~~~~ 

#Correlated Unshuffled (Total) WITHINN
ci_cov_cor_T = stats.t.interval(alpha=0.95, df=len(covar_CORR_T)-1,
              loc=np.mean(covar_CORR_T),
              scale=stats.sem(covar_CORR_T))

#Correlated Shuffled (Pleiotropy) WITHIN
ci_cov_cor_P = stats.t.interval(alpha=0.95, df=len(covar_CORR_P)-1,
              loc=np.mean(covar_CORR_P),
              scale=stats.sem(covar_CORR_P))

#Correlated Delta (LD) WITHIN
ci_cov_cor_Delta = stats.t.interval(alpha=0.95, df=len(covar_CORR_LD)-1,
              loc=np.mean(covar_CORR_LD),
              scale=stats.sem(covar_CORR_LD))


#Uncorrelated Unshuffled (Total)
ci_cov_uncor_T = stats.t.interval(alpha=0.95, df=len(covar_UNCORR_T)-1,
              loc=np.mean(covar_UNCORR_T),
              scale=stats.sem(covar_UNCORR_T))
#Uncorrelated Shuffled (Pleiotropy)
ci_cov_uncor_P = stats.t.interval(alpha=0.95, df=len(covar_UNCORR_P)-1,
              loc=np.mean(covar_UNCORR_P),
              scale=stats.sem(covar_UNCORR_P))
#Uncorrelated Delta (LD)
ci_cov_uncor_Delta = stats.t.interval(alpha=0.95, df=len(covar_UNCORR_LD)-1,
              loc=np.mean(covar_UNCORR_LD),
              scale=stats.sem(covar_UNCORR_LD))

###VARIANCES

#Uncorrelated Unshuffled (Total)
ci_var_uncor_T = stats.t.interval(alpha=0.95, df=len(covar_UNCORR_T)-1,
              loc=np.mean(covar_UNCORR_T),
              scale=stats.sem(covar_UNCORR_T))
#Uncorrelated Shuffled (Pleiotropy)
ci_var_uncor_P = stats.t.interval(alpha=0.95, df=len(covar_UNCORR_P)-1,
              loc=np.mean(covar_UNCORR_P),
              scale=stats.sem(covar_UNCORR_P))
#Uncorrelated Delta (LD)
ci_var_uncor_Delta = stats.t.interval(alpha=0.95, df=len(covar_UNCORR_LD)-1,
              loc=np.mean(covar_UNCORR_LD),
              scale=stats.sem(covar_UNCORR_LD))

#correlated Unshuffled (Total)
ci_var_cor_T = stats.t.interval(alpha=0.95, df=len(var_CORR_T)-1,
              loc=np.mean(var_CORR_T),
              scale=stats.sem(var_CORR_T))
#orrelated Shuffled (Pleiotropy)
ci_var_cor_P = stats.t.interval(alpha=0.95, df=len(var_CORR_P)-1,
              loc=np.mean(var_CORR_P),
              scale=stats.sem(var_CORR_P))
#correlated Delta (LD)
ci_var_cor_Delta = stats.t.interval(alpha=0.95, df=len(var_CORR_LD)-1,
              loc=np.mean(var_CORR_LD),
              scale=stats.sem(var_CORR_LD))


print("---------------- SEX CORRELATED------------------------")
print("Covariance Correlated Unshuffled (Total)")
print("Confidence Intervals")
pprint(ci_cov_cor_T)
print("Mean")
pprint(np.mean(covar_CORR_T))
print("Covariance Correlated Shuffled (Pleiotropy)")
print("Confidence Intervals")
pprint(ci_cov_cor_P)
print("Mean")
pprint(np.mean(covar_CORR_P))
print("Covariance Correlated Delta (LD)")
print("Confidence Intervals")
pprint(ci_cov_cor_Delta)
print("Mean")
pprint(np.mean(covar_CORR_LD))
print("VARIANCE Correlated Unshuffled (Total)")
print("Confidence Intervals")
pprint(ci_var_cor_T)
print("Mean")
pprint(np.mean(var_CORR_T))
print("VARIANCE Correlated Shuffled (PLEIOTROPY)")
print("Confidence Intervals")
pprint(ci_var_cor_P)
print("Mean")
pprint(np.mean(var_CORR_P))
print("VARIANCE Correlated Delta (LD)")
print("Confidence Intervals")
pprint(ci_var_cor_Delta)
print("Mean")
pprint(np.mean(var_CORR_LD))

print("---------------- SEX UNCORRELATED------------------------")
print("Confidence Intervals")
print("Covariance Uncorrelated Unshuffled (Total)")
print("Confidence Intervals")
pprint(ci_cov_uncor_T)
print("Mean")
pprint(np.mean(covar_UNCORR_T))
print("Covariance Uncorrelated Shuffled (Pleiotropy)")
print("Confidence Intervals")
pprint(ci_cov_uncor_P)
print("Mean")
pprint(np.mean(covar_UNCORR_P))
print("Covariance Uncorrelated Unshuffled (LD)")
print("Confidence Intervals")
pprint(ci_cov_uncor_Delta)
print("Mean")
pprint(np.mean(covar_UNCORR_LD))
print("VARIANCE UnCorrelated Unshuffled (Total)")
print("Confidence Intervals")
pprint(ci_var_uncor_T)
print("Mean")
pprint(np.mean(var_UNCORR_T))
print("VARIANCE UnCorrelated Unshuffled (PLEIOTROPY)")
print("Confidence Intervals")
pprint(ci_var_uncor_P)
print("Mean")
pprint(np.mean(var_UNCORR_P))
print("VARIANCE UnCorrelated Unshuffled (LD)")
print("Confidence Intervals")
pprint(ci_var_uncor_Delta)
print("Mean")
pprint(np.mean(var_UNCORR_LD))


print("----------------T TEST--------------------------")

print('COVARIANCE Uncorrelated Versus Correlated')

TT_WT_vs_UT     = scipy.stats.ttest_ind(covar_CORR_T  , covar_UNCORR_T)
TT_WP_vs_UP     = scipy.stats.ttest_ind(covar_CORR_P  , covar_UNCORR_P)
TT_WLD_vs_ULD     = scipy.stats.ttest_ind(covar_CORR_LD  , covar_UNCORR_LD)

print("""
Total Correlated Vs Uncorrelated     : {}
Pleiotropy Correlated Vs Uncorrelated : {}
LD Correlated Vs Uncorrelated     : {}      
      """.format(TT_WT_vs_UT, TT_WP_vs_UP, TT_WLD_vs_ULD))

print('VARIANCE Uncorrelated Versus Correlated')

TT_WT_vs_UT_VAR     = scipy.stats.ttest_ind(var_CORR_T  , var_UNCORR_T)
TT_WP_vs_UP_VAR     = scipy.stats.ttest_ind(var_CORR_P  , var_UNCORR_P)
TT_WLD_vs_ULD_VAR     = scipy.stats.ttest_ind(var_CORR_LD  , var_UNCORR_LD)

print("""
Total Correlated Vs Uncorrelated VAR    : {}
Pleiotropy Correlated Vs Uncorrelated VAR : {}
LD Correlated Vs Uncorrelated   VAR  : {}      
      """.format(TT_WT_vs_UT_VAR, TT_WP_vs_UP_VAR, TT_WLD_vs_ULD_VAR))
