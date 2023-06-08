#MUT VAR A PORTION (weights)
#This script is calculating the Mutational Variance 
#Checked on March 16th 2023

from cgi import test
import os
from pprint import pprint
import sys
from typing import List
import numpy as np
import pandas as pd
import scipy
import scipy.stats as stats
import random
import math

PICKING_MEAN_STD = (0, 0.5)

def t_W_getall(df: pd.Series) -> np.ndarray:
    """get all weights from a timestamp as 1000x4"""
    return np.array(list(map(lambda e: np.fromstring(str(e), sep=',', dtype=float), df.iloc[2:1002])))
def t_C_getall(df: pd.Series) -> np.ndarray:
    """get all contributions from a timestamp as 1000x4x4"""
    return np.array(list(map(lambda _: np.fromstring(str(_), sep=',', dtype=float).reshape((4, 4)), df.iloc[1002:])))

def t_phen(contribs, weights) -> np.ndarray:
    phens_t  = np.array([ np.array(_[0]@_[1].T) for _ in zip(contribs,weights)])
    return phens_t

def phenotype_delta_t(contribs_t, weights_t):
    #sample10k_t = np.random.choice(np.arange(0,NUMBER OF INDIVIDUALS), ANALYSIS, replace=True)
    sample10k_t = np.random.choice(np.arange(0,1000), 1000000, replace=True)
    original_pheno_10k = []
    for sample_i in sample10k_t:
        contrib_i, weight_i = [contribs_t[sample_i], weights_t[sample_i]]
        original_pheno_10k.append(contrib_i@weight_i.T)
    mutated_pheno_10k = []
    for sample_i in sample10k_t:
        mut_contrib, mut_weights = mutation_a(contribs_t[sample_i],weights_t[sample_i])
        mutated_pheno_10k.append(mut_contrib@mut_weights.T)
    #cov_original = np.cov(np.array(original_pheno_10k).T)
    #cov_mutated  = np.cov(np.array(mutated_pheno_10k).T)
    
    delta_t = np.array(mutated_pheno_10k) - np.array(original_pheno_10k)
         
    return [np.array(original_pheno_10k), np.array(delta_t)]

def mutational_variance(contribs_t, weights_t):
        OLD_PHENOS_TP, DELTA_TP = phenotype_delta_t(contribs_t, weights_t)
        DELTA_TP      = np.array(DELTA_TP     )
        OLD_PHENOS_TP = np.array(OLD_PHENOS_TP)

        #%%%%%%%%%%%% MUTATIONAL VARIANCE %%%%%%%%%%%%%
        VAR0_D = np.var(DELTA_TP[:,0])
        VAR1_D = np.var(DELTA_TP[:,1])
        VAR2_D = np.var(DELTA_TP[:,2])
        VAR3_D = np.var(DELTA_TP[:,3])
        
        #Covariances of each trait and its corresponding delta
        COV0_DP = np.cov(DELTA_TP[:,0],OLD_PHENOS_TP[:,0],rowvar=True)[0][1]
        COV1_DP = np.cov(DELTA_TP[:,1],OLD_PHENOS_TP[:,1],rowvar=True)[0][1]
        COV2_DP = np.cov(DELTA_TP[:,2],OLD_PHENOS_TP[:,2],rowvar=True)[0][1]
        COV3_DP = np.cov(DELTA_TP[:,3],OLD_PHENOS_TP[:,3],rowvar=True)[0][1]
        
        #Mutational Variances
        #MUT VARIANCE = VAR(DELTA1) + 2(COV(DELTA1,Z1))
        MUT_VAR0 = VAR0_D + 2*COV0_DP
        MUT_VAR1 = VAR1_D + 2*COV1_DP
        MUT_VAR2 = VAR2_D + 2*COV2_DP
        MUT_VAR3 = VAR3_D + 2*COV3_DP
        
        #%%%%%%%%%%%% MUTATIONAL COVARIANCE %%%%%%%%%%%%%
        
        #Covariance of all deltas
        DELTA_COVMAT = np.cov(DELTA_TP,rowvar=False)
        
        #Covariance of all pairs of traits and deltas 
        COV01_DP = np.cov(DELTA_TP[:,0],OLD_PHENOS_TP[:,1],rowvar=True)[0][1]
        print((np.cov(DELTA_TP[:,0],OLD_PHENOS_TP[:,1],rowvar=True)).shape)
        COV10_DP = np.cov(DELTA_TP[:,1],OLD_PHENOS_TP[:,0],rowvar=True)[0][1]
        COV02_DP = np.cov(DELTA_TP[:,0],OLD_PHENOS_TP[:,2],rowvar=True)[0][1]
        COV20_DP = np.cov(DELTA_TP[:,2],OLD_PHENOS_TP[:,0],rowvar=True)[0][1] 
        COV03_DP = np.cov(DELTA_TP[:,0],OLD_PHENOS_TP[:,3],rowvar=True)[0][1]
        COV30_DP = np.cov(DELTA_TP[:,3],OLD_PHENOS_TP[:,0],rowvar=True)[0][1]
        
        COV12_DP = np.cov(DELTA_TP[:,1],OLD_PHENOS_TP[:,2],rowvar=True)[0][1]
        COV21_DP = np.cov(DELTA_TP[:,2],OLD_PHENOS_TP[:,1],rowvar=True)[0][1]
        COV13_DP = np.cov(DELTA_TP[:,1],OLD_PHENOS_TP[:,3],rowvar=True)[0][1]
        COV31_DP = np.cov(DELTA_TP[:,3],OLD_PHENOS_TP[:,1],rowvar=True)[0][1]
             
        COV23_DP = np.cov(DELTA_TP[:,2],OLD_PHENOS_TP[:,3],rowvar=True)[0][1]
        COV32_DP = np.cov(DELTA_TP[:,3],OLD_PHENOS_TP[:,2],rowvar=True)[0][1]
        
        ##MUT COVARIANCE = COV(Z1,DELTA2) + COV(Z2,DELTA1) + COV(DELTA1,DELTA2)
        MUT_COV01 = COV01_DP + COV10_DP + DELTA_COVMAT[0][1]
        MUT_COV02 = COV02_DP + COV20_DP + DELTA_COVMAT[0][2]
        MUT_COV03 = COV03_DP + COV30_DP + DELTA_COVMAT[0][3]
        MUT_COV12 = COV12_DP + COV21_DP + DELTA_COVMAT[1][2]
        MUT_COV13 = COV13_DP + COV31_DP + DELTA_COVMAT[1][3]
        MUT_COV23 = COV23_DP + COV32_DP + DELTA_COVMAT[2][3]
        
        
        return [MUT_VAR0,MUT_VAR1,MUT_VAR2,MUT_VAR3,MUT_COV01,MUT_COV02,MUT_COV03,MUT_COV12,MUT_COV13,MUT_COV23]
        

def getrep(path: str, repnum: int = -1) -> pd.DataFrame:
    if repnum == -1:
        print("Provide valid replicate number")
        exit(1)
    return pd.read_csv(os.path.join(path, 'replicate{}.csv'.format(repnum)), index_col=False)

def mutation_a(contribs,weights):
    which = random.randint(0, 3)
    mutval = np.random.normal(*(PICKING_MEAN_STD), 1)[0] * 0.003 
    weights[which] = weights[which] + mutval
    return [contribs,weights]

def mutation_c(contribs,weights):
    which = random.randint(0, 16)
    mutval = np.random.normal(*(PICKING_MEAN_STD), 1)[0] * 0.003  
    row = (which)//4
    col = (which) % 4
    contribs[row, col] = contribs[row, col] + mutval
    return [contribs,weights]
    
#This function was written for working with both Contributions (C) and weights (A) at the same time
# def mutation(contribs, weights):
#     sizew = len(weights)
#     which = random.randint(0, 19)
#     mutval = np.random.normal(*(PICKING_MEAN_STD), 1)[0] * 0.003  
#     if which < sizew:
#         weights[which] = weights[which] + mutval
#     else:
#         row = (which-4)//sizew
#         col = (which-4) % sizew
#         contribs[row, col] = contribs[row, col] + mutval

#     return [contribs, weights]

################### Average of all time points of a single replicate ###############
def average_results(ind:pd.DataFrame)->np.ndarray:
    timestamps= []
    count = 0
    for T in  ind.iterrows(): 
        print("Processed T = {}".format(count))
        timestamps.append(mutational_variance(t_C_getall(T[1]), t_W_getall(T[1])))
        count+=1

    MV__BY_REPLICATE = np.array(timestamps).mean(axis=0)
    print("Mutational variances and covariances", MV__BY_REPLICATE)
    return MV__BY_REPLICATE

######################## CALLING DATA ##############################################
corr_cov   = []
uncorr_cov = []

for _i in range(1,501):
    print("COR:Processed replicate {}".format(_i))
    corr_cov.append(average_results(getrep('/scratch/st-whitlock-1/isa/csvs1.1.1',_i)))
    print("UNCOR:Processed replicate {}".format(_i))
    uncorr_cov.append(average_results(getrep('/scratch/st-whitlock-1/isa/csvs1.1.2',_i)))
#####################################################################################

######### Printing Results
# ​

mut_var_CORR = {'m_var_00': corr_cov[:,0], 
                'm_var_11': corr_cov[:,1], 
                'm_var_22': corr_cov[:,2],
                'm_var_33': corr_cov[:,3],
                'm_var_01': corr_cov[:,4],
                'm_var_02': corr_cov[:,5],
                'm_var_03': corr_cov[:,6],
                'm_var_12': corr_cov[:,7],
                'm_var_13': corr_cov[:,8],
                'm_var_23': corr_cov[:,9]}

mut_var_UNCORR = {'m_var_00': uncorr_cov[:,0], 
                'm_var_11': uncorr_cov[:,1], 
                'm_var_22': uncorr_cov[:,2],
                'm_var_33': uncorr_cov[:,3],
                'm_var_01': uncorr_cov[:,4],
                'm_var_02': uncorr_cov[:,5],
                'm_var_03': uncorr_cov[:,6],
                'm_var_12': uncorr_cov[:,7],
                'm_var_13': uncorr_cov[:,8],
                'm_var_23': uncorr_cov[:,9]}
df_corr = pd.DataFrame(mut_var_CORR)
df_uncorr = pd.DataFrame(mut_var_UNCORR)
df_corr.to_csv('A_MutVar_Correlated111.csv', index=False)
df_uncorr.to_csv('A_MutVar_Independent112.csv', index=False)

mean_CORR= np.array(np.mean(corr_cov, axis=0))
mean_UNCORR = np.array(np.mean(uncorr_cov, axis=0))
pprint("averages for correlated")
pprint("Variances")
pprint(mean_CORR[:4])
pprint("Covariances")
pprint(mean_CORR[4:])
pprint("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

pprint("averages for uncorrelated")
pprint("Variances")
pprint(mean_UNCORR[:4])
pprint("Covariances")
pprint(mean_UNCORR[4:])
pprint("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
resultado = scipy.stats.ttest_ind(corr_cov, uncorr_cov)
pprint("p values")
print(resultado)
