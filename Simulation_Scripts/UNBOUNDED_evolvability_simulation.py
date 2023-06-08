
#UNBOUNDED!!!!!

#November 23rd 2022 
#November 1st 2022
#October 23 2022


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
import pickle


def dir_path(string):
    if string == "0":
        return None
    if os.path.isdir(string):
        return string
    else:
        try:
            if not os.path.exists(string):
                os.makedirs(string, exist_ok=True)
                return string
        except:
            raise PermissionError(string)

#the following arguments are required:
# -itstart/--iter_start,
# -itend/--iter_end,
# -ls/--landscape_increment,
# -SP/--shifting_peak, -t/--type

parser = argparse.ArgumentParser(description='Simulation presets')
parser.add_argument("-it"        , "--itern", type=              int     , help    =              "The number of iterations"                                                                                                        )
parser.add_argument('-save'      , '--outdir', type=             dir_path,help     =              ""                                        "Specify the path to write the results of the simulation."""                            )
parser.add_argument("-itstart"   , "--iter_start", type=         int     , required=True, help=   "The number of iterations"                                                                                                        )
parser.add_argument("-itend"     , "--iter_end", type=           int     , required=True, help=   "The number of iterations"                                                                                                        )
parser.add_argument("-ls"        , "--landscape_increment", type=float   , required=True,help=    "Simulation tag for the current instance."                                                                                        )
parser.add_argument("-sim"       , "--siminst", type=            int     , help    =              "Simulation tag for the current instance."                                                                                        )
parser.add_argument("-SP"        , "--shifting_peak", type=      str     , required=True,choices=['correlated'                              , 'uncorrelated', 'block'],help="Flag for whether the fitness landscape changes or not.")
#parser.add_argument('-t'         , '--type', type=               int     , required=True, help=   'Types involved in experiment'                                                                                                    )
parser.add_argument('-initn'     , '--initial_number', type=     int     , help    =              'Starting number of individuals'                                                                                                  )
parser.add_argument('-gpm_rate'  , '--gpmrate', type=            float   , help    =              'GP-map contribution change mutation rate'                                                                                        )
parser.add_argument('-alm_rate'  , '--almrate', type=            float   , help    =              'Allelic mutation rate'                                                                                                           )
parser.add_argument('-res_state_path','--resurrect_state_path',type=str, )                                                                                                                      
parser.add_argument('-res_rep_i','--resurrect_replicate_index',type=int, )                                                                                                                      

args                         = parser.parse_args()
RESURRECT_STATE_PATH         = args.resurrect_state_path
# RESURRECT_REPLICATE_INDEX    = args.resurrect_replicate_index
REPLICATE_N                  = int(args.siminst if args.siminst is not None else 0) # Use this to index into ressurect-state.
BEGIN_DATE                   = datetime.now().strftime("%I:%M%p on %B %d, %Y")
ITSTART                      = int(args.iter_start)
ITEND                        = int(args.iter_end)
GENERATION                   = 1000
ITERATIONS                   = 10000
AMPLITUDE                    = 1
STD                          = 1
TOTAL_fitcount               = []
MUTATION_RATE_CONTRIB_CHANGE = 0
MUTATION_RATE_ALLELE         = 5
PICKING_MEAN_STD             = (0, 0.5)
LOG_EVERY                    = GENERATION*10 #How often do we get data
OUTDIR                       = args.outdir if args.outdir is not None else 0
SHIFTING_FITNESS_PEAK        = args.shifting_peak
LS_INCREMENT                 = float(args.landscape_increment)
LS_SHIFT_EVERY               = int(1e4)
BEGIN_DATE                   = datetime.now().strftime("%I:%M%p on %B %d, %Y")


# with open('state.pickle', 'rb') as handle: 
#     loaded   = pickle.load(handle)
#     contribs = loaded['contribs']
#     optima   = loaded['optima']
#     weights  = loaded['weights']

def make_mutation_plan_alleles(
        _lambda: float,
        period: int = GENERATION):
    """lambda - the rate of the poisson distribution, in this case -- mutrate per generation

	period - number the interval over which to pick, in this case a single generation
	"""

    # ? how many mutations occur in a given period (Generation)
    poolsize = np.random.poisson(_lambda)

    # ? at which iterations do they occur
    iterns = np.random.randint(low=0, high=period, size=poolsize);
    iterns.sort()

    # ? at which positions do they occur?
    entries = random.sample(range(1, (period * 4 + 1)), poolsize);
    entries.sort();
    posns = np.array([*map(lambda x: x % 4, entries)])

    return {
        "posns": posns,
        "iterns": iterns
    }

# Same (Mutation plan) but now for contribution matrix
def make_mutation_plan_contrib(
        _lambda: float,
        period: int = GENERATION):
    """
	@lambda - the rate of the poisson distribution, in this case -- mutrate per generation
	@period - number the interval over which to pick, in this case a single generation
	"""
    # ? how many mutations occur in a given period (Generation)
    poolsize = np.random.poisson(_lambda)

    # ? at which iterations do they occur

    iterns = np.random.randint(low=0, high=period, size=poolsize);
    iterns.sort()

    # ? at which positions do they occur?

    entries = random.sample(range(1, (period * 16 + 1)), poolsize);
    entries.sort();
    posns = np.array([*map(lambda x: ((x % 16) // 4, (x % 16) % 4), entries)])

    return {
        "posns": posns,
        "iterns": iterns
    }


# Actual mutation happening for alleles
def mutate_alleles(alleles: np.ndarray):
    for g in range(alleles.shape[0]):
        if np.random.uniform() <= MUTATION_RATE_ALLELE:
            pick = np.random.normal(*PICKING_MEAN_STD, 1)
            alleles[g] += pick

def get_fitness_map_with_mean(fitness_mean, std):
    def fitness_map(phenotype: np.ndarray):
        return AMPLITUDE * math.exp(-(np.sum(((phenotype - fitness_mean) ** 2) / (2 * std ** 2))))
    return fitness_map
    
# everything together
class Universe:

    def __init__(
            self,
            current_iter: int,
            ALLS: np.ndarray,
            GPMS: np.ndarray,
            PHNS: np.ndarray,
            mean: np.ndarray,
    ) -> None:

        # ? ------------------------------ [ STATE ]
        self.it = current_iter
        self.ALLS = ALLS
        self.GPMS = GPMS
        self.PHNS = PHNS

        # ? ------------------------------ [ ENV ]
        self.mean = mean
        #self.mutation_plan_contrib = make_mutation_plan_contrib(MUTATION_RATE_CONTRIB_CHANGE)
        self.mutation_plan_alleles = make_mutation_plan_alleles(MUTATION_RATE_ALLELE)

        # ? ------------------------------ [ AGtrs ]
        # What I am saving into the file
        self.agg = {
              "began_logging_at"      : -1,
              "logged_every"         : LOG_EVERY,
              # 'covar_slices'         : np.array([np.cov(self.PHNS.T)]),
              "weight_vectors"       : [],
              "contribution_matrices": [],
              "position_optima"      : [],
              "all_fitness"          : [],
              "elapsed"              : 0
        }
        # self.fitmean_agg = np.array([])

    # def save_state(self):
    #     state = {
    #         "last_iteration": self.it,
    #         "alleles": self.ALLS,
    #         "gpms": self.GPMS,
    #         "fitness_mean": self.mean
    #     }
    #
    #     with open(os.path.join(OUTDIR, 'state_{}.pkl'.format(REPLICATE_N)), 'wb') as f: pickle.dump(state, f)

    def save_aggregate(self)->None:

        OUTPUT_PATH = os.path.join(OUTDIR, "popinfo_replicate{}.pickle".format(REPLICATE_N))

        with open(OUTPUT_PATH, 'wb') as outfile:
            pickle.dump(self.agg, outfile, protocol=pickle.HIGHEST_PROTOCOL)

        # with open(os.path.join(OUTDIR, "popinfo_replicate{}.json".format(REPLICATE_N)), 'w') as outfile:
        #     json.dump(self.agg, outfile)
        #     print("Wrote to {}.".format(outfile))

    #birth
    def pick_parent(self) -> int:

        indices = np.arange(len(self.PHNS))
        fitnesses = np.array([*map(get_fitness_map_with_mean(self.mean,STD), self.PHNS)])
        cumulativefit = reduce(lambda x, y: x + y, fitnesses)

        return np.random.choice(indices, p=fitnesses / cumulativefit)


    #Function related to Death
    def pick_death(self) -> int:
        indices = np.arange(len(self.PHNS))
        return np.random.choice(indices)

    #Birth-death process: what happens in a iteration.
    def birth_death(self):

        # ! Below is a logic for "Record every $LOG_EVERY iterations _in the last 10% of the duration of the current run."
        # if self.it > ( ITEND - (math.ceil((ITEND - ITSTART) / 10)) ) and not (self.it % LOG_EVERY):

        # ! Below is a logic for "Record every $LOG_EVERY iterations "
        if not (self.it % LOG_EVERY):
            print("Recorded state at iteration {}".format(self.it))

            if self.agg['began_logging_at'] == -1:
                self.agg['began_logging_at'] = ITSTART

            self.agg["weight_vectors"]        += self.ALLS.tolist()
            self.agg["contribution_matrices"] += self.GPMS.tolist()
            self.agg['position_optima']       += self.mean.tolist()
            self.agg['all_fitness']           += np.array([*map(get_fitness_map_with_mean(self.mean,STD), self.PHNS)]).tolist()
            self.agg['elapsed']               += LOG_EVERY # How many iterations passed


        if (not self.it % LS_SHIFT_EVERY):
            print("Shifted landscape at iteration {}".format(self.it))
            self.shift_landscape(LS_INCREMENT, SHIFTING_FITNESS_PEAK)

        death_index = self.pick_death()
        birth_index = self.pick_parent()

        _alleles = np.copy(self.ALLS[birth_index])
        _contribs = np.copy(self.GPMS[birth_index])
        # Checking if according to mutation plan whether there is a mutation happening at this birth
        while bool(len(self.mutation_plan_alleles['iterns'])) and self.it % GENERATION == self.mutation_plan_alleles['iterns'][0]:
            posn = self.mutation_plan_alleles['posns'][0]
            _alleles[posn] += np.random.normal(*PICKING_MEAN_STD)
            self.mutation_plan_alleles['iterns'] = self.mutation_plan_alleles['iterns'][1:]
            self.mutation_plan_alleles['posns'] = self.mutation_plan_alleles['posns'][1:]
            
        ###NO MUTATION ON CONTRIBUTION FOR THIS TYPE OF SIMULATIONS
        # while bool(len(self.mutation_plan_contrib['iterns'])) and self.it % GENERATION == \
        #         self.mutation_plan_contrib['iterns'][0]:
        #     posn = tuple(self.mutation_plan_contrib['posns'][0])
        #     _contribs[posn] += np.random.normal(*PICKING_MEAN_STD)
        #     self.mutation_plan_contrib['iterns'] = self.mutation_plan_contrib['iterns'][1:]
        #     self.mutation_plan_contrib['posns'] = self.mutation_plan_contrib['posns'][1:]

        if self.it % GENERATION == 0:
            #self.mutation_plan_contrib = make_mutation_plan_contrib(MUTATION_RATE_CONTRIB_CHANGE)
            self.mutation_plan_alleles = make_mutation_plan_alleles(MUTATION_RATE_ALLELE)

        self.PHNS[death_index] = _contribs @ _alleles.T
        self.ALLS[death_index] = _alleles
        self.GPMS[death_index] = _contribs
        self.it += 1

    def get_avg_fitness(self) -> float:
        return reduce(lambda x, y: x + y, map(get_fitness_map_with_mean(self.mean,STD), self.PHNS)) / len(self.PHNS)

    # def write_fitness_data(self):
    #     with open(os.path.join(OUTDIR, 'fitness_data', 'data{}.pkl'.format(REPLICATE_N)), 'wb') as f: pickle.dump(
    #         self.fitmean_agg, f)

    # def write_covar_pkl(self):
    #
    #     outfile = os.path.join(OUTDIR, 'var_covar', 'mean_var_covar_{}.pkl'.format(REPLICATE_N))
    #     _ = {
    #         'covar_slices': self.covar_agg['covar_slices'],
    #         'elapsed': self.covar_agg['elapsed'],
    #         'began_loggin_at': self.covar_agg['began_loggin_at'],
    #         'logged_every': self.covar_agg['logged_every']
    #     }
    #     with open(outfile, 'wb') as log:
    #         pickle.dump(_, log)
    #
    # def log_var_covar(self):
    #
    #     self.covar_agg['covar_slices'] = np.row_stack([self.covar_agg['covar_slices'], np.array([np.cov(self.PHNS.T)])])
    #     self.covar_agg['elapsed'] += 1
    #     if self.covar_agg['began_loggin_at'] == -1:
    #         self.covar_agg['began_loggin_at'] = self.it

    def shift_landscape(
            self,
            LANDSCAPE_INCREMENT: float,
            CORRELATED: bool) -> None:
        # ? Correlated shifts
        print(self.mean)
        if CORRELATED == 'correlated':
            self.mean += np.random.choice([1, -1]) * LANDSCAPE_INCREMENT
        # ? Uncorrelated shifts
        elif CORRELATED == 'uncorrelated':
           
            # *Small IT
           
            for i, x in enumerate(self.mean):
                coin = np.random.choice([1, -1])
                self.mean[i] += coin * LANDSCAPE_INCREMENT
        elif CORRELATED == 'block':
                for pair_inds in [[0, 1], [2, 3]]:
                    coin = np.random.choice([1, -1])
                    self.mean[pair_inds] += coin * LANDSCAPE_INCREMENT
        else:
            exit("Unspecified landscape shift type")


def print_receipt() -> None:
    receipt = {
          "ITSTART"                     : ITSTART,
          "ITEND"                       : ITEND,
          "REPLICATE_N"                 : REPLICATE_N,
          "OUTDIR"                      : OUTDIR,
        # "RESSURECTED"                 : RESSURECT,
        # "INDTYPE"                     : INDTYPE,
        # "POPN"                        : POPN,
          "SHIFTING_FITNESS_PEAK"       : SHIFTING_FITNESS_PEAK,
          "LS_INCREMENT"                : LS_INCREMENT,
          "MUTATION_RATE_ALLELE"        : MUTATION_RATE_ALLELE,
          "MUTATION_RATE_CONTRIB_CHANGE": MUTATION_RATE_CONTRIB_CHANGE,
          "LS_SHIFT_EVERY"              : LS_SHIFT_EVERY,
          "STD"                         : STD,
          "AMPLITUDE"                   : AMPLITUDE,
          "LOG_EVERY"                   : LOG_EVERY,
          "date_finished"               : datetime.now().strftime("%I:%M%p on %B %d, %Y"),
          "date_started"                : BEGIN_DATE,
          "PICKING_MEAN_STD"            : [*PICKING_MEAN_STD]
    }
    with open(os.path.join(OUTDIR, "parameters_replicate{}.json".format(REPLICATE_N)), 'w') as infile:
        json.dump(receipt, infile)

# This is not updaated
def ressurect_at_index(replicate_index,state_file_path):
    with open(state_file_path, 'rb') as inf:
        state      = pickle.load(inf)
        contribs   = state['contribs'][replicate_index]
        weights    = state['weights' ][replicate_index]
        optima     = state['optima'  ][replicate_index]
        phenotypes = []
        for i in range(len(contribs)):
            phenotypes.append(contribs[i] @weights[i].T)
        phenotypes = np.array(phenotypes)
        print(phenotypes.shape)
        if ITEND <= ITSTART:
            print(f"End iteration that was specified {ITEND} is lower than this population's 'age'({ITSTART}). Exited ")
            exit(1)

        return [Universe(ITSTART,
                         weights,
                         contribs,
                         phenotypes,
                         optima),ITSTART]


# file_name_last_it_C = "/Users/idoo/code_covar_evo/Data_Covar/exp111-allsameC.pickle"
#This is the file with the last iteration of a simulation where we changed the "C" to be all the same
u:Universe;
[u, start_iter] = ressurect_at_index(REPLICATE_N,RESURRECT_STATE_PATH)
ITSTART         = start_iter
for it in range(ITSTART, ITEND + 1): 
    u.birth_death()
u.save_aggregate()
    


