#Preparing simulation experiment - No mutation on C, with mutation on weights, after it has evolved for a number of iterations

import sys, os
import numpy as np
import pandas as pd
from tempfile import TemporaryFile
import pickle

my_path_pickle = '/Users/idoo/code_covar_evo/Data_Covar/exp112-allsameC.pickle' 
# def mode_rows(a): ###Should be for 3D not for 2D!!
#     a = np.ascontiguousarray(a)
#     void_dt = np.dtype((np.void, a.dtype.itemsize * np.prod(a.shape[1:])))
#     _,ids, count = np.unique(a.view(void_dt).ravel(), \
#                                 return_index=1,return_counts=1)
#     largest_count_id = ids[count.argmax()]
#     most_frequent_row = a[largest_count_id]
#     return most_frequent_row

def most_common(arr2d):
    import operator
    hashed = list(map(lambda m2x2: (hash(np.array2string(m2x2)), m2x2), arr2d))
    # `hashed` is a list of tuples, where the first element is the hash of the corresponding matrix
    dict = {}
    # tally them up using hashes as keys
    for (h, mat) in hashed:
        dict[h] = (1, mat) if h not in dict else (dict[h][0]+1, mat)
    # `sorted` is an array of keyvalue pairs (count, matrix) in descending order
    return sorted(dict.values(), key=operator.itemgetter(0), reverse=True)[0][1]


outfile        = TemporaryFile()
lolo           = TemporaryFile()
LastIterations = []
weights_LastIt = []
contrib_LastIt = []
optima_LastIt  = []

# ADD_MAP = np.array([0.2,0.2,0.2,0.2])
# #Correlated = [0.2,0.2,0.2,0.2]
# #Block C = [0.2,0.2,-0.2,-0.2]
# #Independent = [-0.2,0.2,-0.2,0.2]

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
#source_folder = "exp1.1.1" 

for _i in range(500):
    print("Processed replicate {}".format(_i))
    try:
        temporario = getrep('/Users/idoo/code_covar_evo/Data_Covar/exp1.1.2/csvs',_i)
        #LastIterations.append(temporario.iloc[-1])
        contribs = t_C_getall(temporario) #grab all contribs
        most_common_C = most_common(contribs) #find most common contrib
        N_ind = contribs.shape[1] #find size of population
        New_pop_C = np.tile([most_common_C], (N_ind,1)) #create a population with only the most common contrib
         
        weights = t_W_getall(temporario)
        optima = t_optima(temporario)
        
        weights_LastIt.append(t_W_getall(temporario))
        contrib_LastIt.append(t_C_getall(temporario))
        optima_LastIt.append(t_optima(temporario))
    except:
        ...
        
contrib_LastIt = np.array(contrib_LastIt)
weights_LastIt = np.array(weights_LastIt)
optima_LastIt = np.array(optima_LastIt) 

state = {
    'contribs': contrib_LastIt,
    'weights' : weights_LastIt,
    'optima'  : optima_LastIt
     }

with open(my_path_pickle, 'wb') as handle:
    pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
