## THIS SCRIPT GRABS THE LAST ITERATION OF A SIMULATION AND GETS IT READY FOR A NEW SIMULATION
##June 29 22

import sys, os
import numpy as np
import pandas as pd
from tempfile import TemporaryFile

outfile        = TemporaryFile()
lolo           = TemporaryFile()
LastIterations = []
weights_LastIt = []
contrib_LastIt = []
optima_LastIt  = []

################ -- #################
def t_W_getall(df: pd.Series) -> np.ndarray:    
    """get all weights from a timestamp as 1000x4"""
    return np.array(list(map(lambda e: np.fromstring(str(e), sep=',', dtype=float), df.iloc[-1,2:1002])))

def t_C_getall(df: pd.Series) -> np.ndarray:
    """get all contributions from a timestamp as 1000x4x4"""
    return np.array(list(map(lambda _: np.fromstring(str(_), sep=',', dtype=float).reshape((4, 4)), df.iloc[-1,1002:])))

def t_optima(df: pd.Series) -> np.ndarray:
    """get all optima from a timestamp as 1000x4"""
    return np.fromstring(df.iloc[-1,1], sep=',',dtype=np.float64)
    
def getrep(path: str, repnum: int = -1) -> pd.DataFrame:
    if repnum == -1:
        print("Provide valid replicate number")
        exit(1)
    return pd.read_csv(os.path.join(path, 'replicate{}.csv'.format(repnum)), index_col=False)
#####################################################################################################################

EXP_NUMBER    = 3

for _i in range(1,500):
    print("Processed replicate {}".format(_i))
    try:
        temporario = getrep(f'/Users/idoo/code_covar_evo/Data_Covar/exp3.1.1/csvs',_i)
        weights_LastIt.append(t_W_getall(temporario))
        contrib_LastIt.append(t_C_getall(temporario))
        optima_LastIt.append(t_optima(temporario))
    except Exception as e: 
        print(e); ...

        
contrib_LastIt = np.array(contrib_LastIt)
weights_LastIt = np.array(weights_LastIt)
optima_LastIt  = np.array(optima_LastIt)


with open(f'test_weights{EXP_NUMBER}.npy', 'wb') as fw:   
    np.save(fw, weights_LastIt, allow_pickle = True)
    
with open(f'test_contrib{EXP_NUMBER}.npy', 'wb') as fc:   
    np.save(fc, contrib_LastIt, allow_pickle = True)

with open(f'test_optima{EXP_NUMBER}.npy', 'wb') as fo:
    # print("saving optima ", optima_LastIt)
    np.save(fo, optima_LastIt, allow_pickle = True)
