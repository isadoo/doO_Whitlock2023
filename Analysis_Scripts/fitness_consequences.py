#This is supposed to create the CSV of fitness change per time for all experiments
#this csv will be used to create the graphs.
#It takes the pickled/json files from the simulations and creates the CSVs 

#November 19 2022
#Last checked: Mar 2023


import sys, os
import numpy as np
import pandas as pd
from tempfile import TemporaryFile
import pickle
import argparse


parser = argparse.ArgumentParser(description='Simulation presets')
parser.add_argument("-EXP_NAME", "--experiment_name", type=str, required=True,choices=
                    ['correlated_111','correlated_112','correlated_311', 'uncorrelated_111',
                     'uncorrelated_112','uncorrelated_311', 'block_111','block_112','block_311'])
parser.add_argument("-EXP_P", "--experiment_path", type=str, required=True)
args = parser.parse_args()
EXPERIMENT_NAME = args.experiment_name
EXPERIMENT_PATH = args.experiment_path


def getrep(state_file_path):
    with open(state_file_path, 'rb') as inf:
        return pickle.load(inf) #this is the "replicate" dictionary == "self.agg" in evolvability_simulation.py

def rep_fitness(replicate):
    by_timestamp = np.array_split(replicate["all_fitness"],501)
    #print("print partitioned by a 1000", len(by_timestamp))
    mean_fit = np.array([
        *map(lambda x: np.mean(x), by_timestamp) #mean over the population
        ])
    
    return mean_fit #vector of fitness of each time point of a replicate
   
   # we loop through all the popinfo_replicate{}.pickle in a folder to get the mean fitness through time of each replicate

ALL_REPS_fit = []
for _i in range(500):
    #print("Processed replicate {}".format(_i))
    #current_replicate = getrep(f'/Users/idoo/code_covar_evo/Evolvability_Results/{EXPERIMENT_NAME}/popinfo_replicate{_i}.pickle')
    try:
        current_replicate = getrep(f'{EXPERIMENT_PATH}/popinfo_replicate{_i}.pickle')
        ALL_REPS_fit.append(rep_fitness(current_replicate))
    #print("Shape of ALL_REPS_fit")
    #print(np.array(ALL_REPS_fit).shape)
    except:
        print(f"sorry, didn't find {_i}")

MEAN_ALL_REPS_fit = np.mean(np.array(ALL_REPS_fit),0)
#print("fitness mean shape")
print(MEAN_ALL_REPS_fit.shape)


timepoints = np.arange(0,len(MEAN_ALL_REPS_fit),1,dtype=int)
title_col = (f'fitness_{EXPERIMENT_NAME}')
df = pd.DataFrame({'time':timepoints, title_col:MEAN_ALL_REPS_fit})
df.to_csv((f'{EXPERIMENT_NAME}_fitnesstime.csv'),index=False)
