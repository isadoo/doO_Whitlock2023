#measuring additive variance and covariance for the sexual populations
#August 22
import random
import timeit
from datetime import datetime
from functools import reduce
import json
from time import time
import  os
from tracemalloc import start
import numpy as np
import math
import argparse
import pandas as pd
from scipy import stats

n_couples = 1000 


def t_W_getall(df: pd.Series) -> np.ndarray:
    """get all weights from a timestamp as 1000x4"""
    return np.array(list(map(lambda e: np.fromstring(str(e), sep=',', dtype=float), df.iloc[2:1002])))
def t_C_getall(df: pd.Series) -> np.ndarray:
    """get all contributions from a timestamp as 1000x4x4"""
    return np.array(list(map(lambda _: np.fromstring(str(_), sep=',', dtype=float).reshape((4, 4)), df.iloc[1002:])))

def getrep(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=False)

def AdditiveVariance(weights,contribs):
    #Making random couples
    midparents = []
    offsprings = []
    ngenesC = np.size(contribs[1])
    ngenesW = np.size(weights[1])

    for m in range(n_couples):
        female, male = [random.randrange(0,len(weights)) for p in range(0,2)]
        #getting the information from the parents
        motherw = np.copy(weights[female,:])
        fatherw = np.copy(weights[male])
        motherc = np.copy(contribs[female]) 
        fatherc = np.copy(contribs[male])
        #getting the phenotype of the parents and saving the avg 
        phenmom = motherc @ motherw.T
        phendad = fatherc @ fatherw.T
        midparents.append((phenmom + phendad)/2)
        
        #Choosing genes from one parent or another
        mask = np.random.choice([0,1], size=(20,)).reshape((5,4))
        motherw[mask[ :1][0]==1] = fatherw[mask[ :1][0]==1]
        motherc[mask[1: ]==1] = fatherc[mask[1: ]==1]
        #Add offspring to list
        offsprings.append(motherc @ motherw.T)
    
    midparents = np.array(midparents)
    meanmidparents = np.mean(midparents, axis = 1)
    offsprings = np.array(offsprings)
    meanoffsprings =  np.mean(offsprings, axis = 1)
    #Regression to calculate heritability
    resultvar = stats.linregress(meanmidparents,meanoffsprings)
    heritability = resultvar.slope
    Variance_Parents = np.var(meanmidparents) 
    Var = heritability*Variance_Parents
    
    #Covariance
    
    #for t in range(3):    
    #    resultcov = stats.linregress(midparents[:,t],offsprings[:,t+1])
    #    hcov = resultcov.slope
    #    var_pertrait = np.var(midparents[:,t])
    #    cov.append(hcov*var_pertrait)
    res01 = stats.linregress(midparents[:,0],offsprings[:,1])
    res10 = stats.linregress(midparents[:,1],offsprings[:,0])
    
    res02 = stats.linregress(midparents[:,0],offsprings[:,2])
    res20 = stats.linregress(midparents[:,2],offsprings[:,0])
    
    res03 = stats.linregress(midparents[:,0],offsprings[:,3])
    res30 = stats.linregress(midparents[:,3],offsprings[:,0])
    
    res12 = stats.linregress(midparents[:,1],offsprings[:,2])
    res21 = stats.linregress(midparents[:,2],offsprings[:,1])
    
    res13 = stats.linregress(midparents[:,1],offsprings[:,3])
    res31 = stats.linregress(midparents[:,3],offsprings[:,1])
    
    res23 = stats.linregress(midparents[:,2],offsprings[:,3])
    res32 = stats.linregress(midparents[:,3],offsprings[:,2])
    
    var_0 = np.var(midparents[:,0])
    var_1 = np.var(midparents[:,1])
    var_2 = np.var(midparents[:,2])
    var_3 = np.var(midparents[:,3])
    
    cov01 = var_0*res01.slope
    cov10 = var_1*res10.slope
    COV01 = (cov01 + cov10)/2
    cov02 = var_0*res02.slope
    cov20 = var_2*res20.slope
    COV02 = (cov02 + cov20)/2
    cov03 = var_0*res03.slope
    cov30 = var_3*res30.slope
    COV03 = (cov03 + cov30)/2
    cov12 = var_1*res12.slope
    cov21 = var_2*res21.slope
    COV12 = (cov12 + cov21)/2
    cov13 = var_1*res13.slope
    cov31 = var_3*res31.slope
    COV13 = (cov13 + cov31)/2
    cov23 = var_2*res23.slope
    cov32 = var_3*res32.slope
    COV23 = (cov23 + cov32)/2
    Cov = (COV01+COV02+COV03+COV12+COV13+COV23)/6
    return Var,Cov
        

Var = []
Covar = []
#contribs=[]
#weights = np.empty((0,4), float)
#contribs = np.empty((4,4),float)
for _i in range(1,500):
    print("Processed replicate {}".format(_i))
    try:  
        #temporario = getrep(f'/Users/idoo/code_covar_evo/Data_Covar/exp2.1.1/csvs/replicate{_i}.csv')
        temporario = getrep(f'/scratch/st-whitlock-1/isa/csvs2.1.1/replicate{_i}.csv')
    except:
        print("bububu")
        ...
    r_Var = []
    r_Covar =[]
    for T in temporario.iterrows():
        weights = t_W_getall(T[1])
        contribs = t_C_getall(T[1])
        var,covar = (AdditiveVariance(weights,contribs))
        r_Var.append(var)
        r_Covar.append(covar)
    r_Var = np.array(r_Var)
    r_Covar = np.array(r_Covar)
    mean_r_var = np.mean(r_Var)
    mean_r_covar = np.mean(r_Covar)
    Var.append(mean_r_var)
    Covar.append(mean_r_covar)

Var = np.array(Var)
Covar = np.array(Covar)
Mean_Var = np.nanmean(Var)
Mean_Covar = np.nanmean(Covar)
print("Total mean Covar")
print(Mean_Covar)
ADDITIVE_SEXCorrelated_211 = pd.DataFrame({'Environment': np.full(len(Covar), "Correlated_Sexual"),
                'Variance_Additive': Var,
                'Covariance_Additive': Covar,
                   })


ADDITIVE_SEXCorrelated_211.to_csv('ADDITIVE_SEX_Correlated_211.csv')

Ind_Var = []
Ind_Covar = []

for _h in range(1,500):
    print("Processed replicate {}".format(_h))
    try:
        #temporario = getrep(f'/Users/idoo/code_covar_evo/Data_Covar/exp2.1.2/csvs',_h)
        temporario = getrep(f'/scratch/st-whitlock-1/isa/csvs2.1.2/replicate{_h}.csv')
    except:
        ...
    r_Var_I = []
    r_Covar_I =[]
    for T_ in  temporario.iterrows():
        weights_I = t_W_getall(T_[1])
        contribs_I = t_C_getall(T_[1])
        var_I,covar_I = (AdditiveVariance(weights_I,contribs_I))
        r_Var_I.append(var_I)
        r_Covar_I.append(covar_I)
    r_Var_I = np.array(r_Var_I)
    r_Covar_I = np.array(r_Covar_I)
    mean_r_var_I = np.mean(r_Var_I)
    mean_r_covar_I = np.mean(r_Covar_I)
    Ind_Var.append(mean_r_var_I)
    Ind_Covar.append(mean_r_covar_I)

Ind_Var = np.array(Ind_Var)
Ind_Covar = np.array(Ind_Covar)
I_Mean_Var = np.mean(Ind_Var)
I_Mean_Covar = np.mean(Ind_Covar)

ADDITIVE_SEXUncorrelated_212 = pd.DataFrame({'Environment': np.full(len(Ind_Covar), "Uncorrelated_Sexual"),
                                 'Variance_Additive': Ind_Var,
                                'Covariance_Additive': Ind_Covar,
                   })


ADDITIVE_SEXUncorrelated_212.to_csv('ADDITIVE_SEX_Uncorrelated_212.csv')

print("Correlated\n")
print("VAR")
print(Mean_Var)
#contribs = np.array(contribs)
print("------------\n") 
#print(weights[1,:]) #A single individual 
print("COVAR")
print(Mean_Covar)
print("------------------------------------\n") 

print("Independent\n") 
print("VAR") 
print(I_Mean_Var)
print('-------\n')
print("COVAR")
print(I_Mean_Covar)

print("------------------------------------\n") 
print("Variance t test\n") 
print(stats.ttest_ind(Var,Ind_Var))
print("Covariance t test\n")
print(stats.ttest_ind(Covar,Ind_Covar))

ci_var_Independent = stats.t.interval(alpha=0.95, df=len(Ind_Var)-1,
              loc=np.mean(Ind_Var),
              scale=stats.sem(Ind_Var))

ci_var_Correlated = stats.t.interval(alpha=0.95, df=len(Var)-1,
              loc=np.mean(Var),
              scale=stats.sem(Var))

ci_covar_Independent = stats.t.interval(alpha=0.95, df=len(Ind_Covar)-1,
              loc=np.mean(Ind_Var),
              scale=stats.sem(Ind_Var))

ci_covar_Correlated = stats.t.interval(alpha=0.95, df=len(Covar)-1,
              loc=np.mean(Var),
              scale=stats.sem(Var))



print("Confidence intervals")
print("Var Correlated")
print(ci_var_Correlated)
print("Var Indepednent")
print(ci_var_Independent)
print("Covar Correlated")
print(ci_covar_Correlated)
print("Covar Independent")
print(ci_covar_Independent)

