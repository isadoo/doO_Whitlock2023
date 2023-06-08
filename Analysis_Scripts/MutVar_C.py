#MUT VAR A PORTION (weights)
#This script is calculating the Mutational Variance 
#Checked on March 19th 2023

import os
import random
import math
import itertools


import numpy as np
import pandas as pd
import scipy.stats as stats
import inspect

from cgi import test
from pprint import pprint
import concurrent.futures

# ※ ------------------------------------------------- INITS ------------------------------------------------ ※
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

# ※ -------------------------------------------------- ※ -------------------------------------------------- ※

def mutation_c(contribs):
    which = random.randint(0, 16)
    mutval = np.random.normal(*(PICKING_MEAN_STD), 1)[0] * 0.003  
    row = (which)//4
    col = (which) % 4
    contribs[row, col] = contribs[row, col] + mutval
    return np.array(contribs)

def phenotype_delta_t(contribs_t, weights_t, sample_N:int=1000000):
    index_pool       = np.random.choice(np.arange(0,1000), sample_N, replace=True)

    sampled_pheno     = []
    sampled_pheno_mut = []

    contribs_sampled  = contribs_t[index_pool]
    weights_sampled  = weights_t[index_pool]

    mut_contrib = mutation_c(contribs_t[index_pool])

    sampled_pheno     = [ contrib_i@weight_i.T for contrib_i, weight_i in zip(contribs_sampled, weights_sampled)]
    sampled_pheno_mut = [ contrib_i@weight_i_mut.T for contrib_i, weight_i_mut in zip(mut_contrib,weights_sampled )]

    delta_t = np.array(np.array(sampled_pheno_mut) - np.array( sampled_pheno ))
         
    return [sampled_pheno, delta_t]


def mutational_variance(contribs_t, weights_t):
    print("[{}]: Calculating mutvar".format(inspect.currentframe().f_code.co_name))
    OLD_PHENOS_TP, DELTA_TP = phenotype_delta_t(contribs_t, weights_t)

    DELTA_TP      = np.array(DELTA_TP)
    OLD_PHENOS_TP = np.array(OLD_PHENOS_TP)
    
    # Calculate variances and covariances
    var_cov_dict   = {}
    covar_cov_dict = {}


    for i in range(4):
        var_cov_dict[f'VAR{i}_D'] = np.var(DELTA_TP[:,i])
        for j in range(4):
            key = f'COV{i}{j}_DP'
            #COV{i}{j}_DP means covariance of delta i with old phenotype j 
            covar_cov_dict[key] = np.cov(DELTA_TP[:,i], OLD_PHENOS_TP[:,j], rowvar=True)[0][1]
    
    # Calculate mutational variances
    mut_var_dict = {}
    for i in range(4):
        key = f'MUT_VAR{i}'
        mut_var_dict[key] = var_cov_dict[f'VAR{i}_D'] + 2*covar_cov_dict[f'COV{i}{i}_DP']
    
    # Calculate mutational covariances
    mut_cov_dict = {}
    delta_covmat = np.cov(DELTA_TP,rowvar=False)
    for i, j in itertools.combinations(range(4), 2):
        if i == j:
            continue
        else:
            key = f'MUT_COV{i}{j}'
            mut_cov_dict[key] = covar_cov_dict[f'COV{i}{j}_DP'] + covar_cov_dict[f'COV{j}{i}_DP'] + delta_covmat[i][j]

    # Combine variances and covariances into a list
    mut_var_list = list(mut_var_dict.values())
    mut_cov_list = list(mut_cov_dict.values())
    
    # Return the result
    return mut_var_list + mut_cov_list

def getrep(path: str, repnum: int = -1) -> pd.DataFrame:
    if repnum == -1:
        print("Provide valid replicate number")
        exit(1)
    return pd.read_csv(os.path.join(path, 'replicate{}.csv'.format(repnum)), index_col=False)



################### Average of all time points of a single replicate ###############
def pool_average_results(ind:pd.DataFrame, threads_n:int)->np.ndarray:
    timestamps = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=threads_n) as executor:
        # Submit a thread for each piece of data
        future_results = {
            executor.submit(mutational_variance, t_C_getall(row) , t_W_getall(row)) : row for _, row in ind.iterrows()
            }
        timestamps        = [future.result() for future in concurrent.futures.as_completed(future_results)]

    MV__BY_REPLICATE = np.array(timestamps).mean(axis=0)
    print("Got: Mutational variances and covariances MV_BY_REP : {}".format(MV__BY_REPLICATE.shape))
    return np.array(MV__BY_REPLICATE)

######################## CALLING DATA ##############################################
corr_cov   = []
uncorr_cov = []
N_THREADS  = 10


# TODO: can make this a replicate-wise thread pool 
for _i in range(1,501):
    print("Ingress: correlated replicate {}".format(_i))
    corr_cov.append(pool_average_results(getrep('/scratch/st-whitlock-1/isa/csvs1.1.1',_i), N_THREADS))

    print("Ingress: uncorrelated replicate {}".format(_i))
    uncorr_cov.append(pool_average_results(getrep('/scratch/st-whitlock-1/isa/csvs1.1.2',_i),N_THREADS))
    #####################################################################################

######### Printing Results
corr_cov = np.array(corr_cov)
uncorr_cov = np.array(uncorr_cov)

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
#####################################################################################

######### Printing Results
# ​

# mut_var_CORR = {f'm_var_{ii}{jj}': corr_cov[:,jj] for jj, ii in enumerate(range(4) * 3)}
# mut_var_UNCORR = {f'm_var_{nn}{mm}': uncorr_cov[:,mm] for mm, nn in enumerate(range(4) * 3)}

df_corr   = pd.DataFrame(mut_var_CORR)
df_uncorr = pd.DataFrame(mut_var_UNCORR)

df_corr.to_csv('C_MutVar_Correlated111NEW.csv', index=False)
df_uncorr.to_csv('C_MutVar_Independent112NEW.csv', index=False)

mean_CORR   = np.array(np.mean(corr_cov, axis=0))
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
resultado = stats.ttest_ind(corr_cov, uncorr_cov)
pprint("p values")
print(resultado)