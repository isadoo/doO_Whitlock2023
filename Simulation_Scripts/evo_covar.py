import pprint
import random
import timeit
from datetime import datetime
from functools import reduce
import json
from time import time
import sys, os
from typing import Callable, List
import numpy as np
import math
import argparse
import pickle

# This is to check if the files are going to an existing place.
# Checking if the string that is passed on is a directory
# If string doesn't exist as directory, create one.
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

# These are all the arguments for the script
# argparse lets scripts take arguments from command line
parser = argparse.ArgumentParser(description='Simulation presets')
parser.add_argument("-it", "--itern", type=int, help="The number of iterations")
parser.add_argument('-save', '--outdir', type=dir_path,
                    help=""                                                                                           "Specify the path to write the results of the simulation.""")
parser.add_argument("-itstart", "--iter_start", type=int, required=True, help="The number of iterations")
parser.add_argument("-itend", "--iter_end", type=int, required=True, help="The number of iterations")
parser.add_argument("-ls", "--landscape_increment", type=float, required=True,
                    help="Simulation tag for the current instance.")
parser.add_argument("-sim", "--siminst", type=int, help="Simulation tag for the current instance.")
parser.add_argument("-SP", "--shifting_peak", type=str, required=True,
                    choices=['correlated', 'uncorrelated', 'pairwise', '8_bimodular','8_quadmodular'],
                    help="Flag for whether the fitness landscape changes or not.")
parser.add_argument('-t', '--type', type=int, required=True, help='Types involved in experiment')
parser.add_argument('-initn', '--initial_number', type=int, help='Starting number of individuals')
parser.add_argument('-gpm_rate', '--gpmrate', type=float, help='GP-map contribution change mutation rate')
parser.add_argument('-alm_rate', '--almrate', type=float, help='Allelic mutation rate')
parser.add_argument('--resurrect', action='store_true')

# Parameters for simulation
args                         = parser.parse_args()
GENERATION                   = 1000
ITSTART                      = int(args.iter_start)
ITEND                        = int(args.iter_end)
REPLICATE_N                  = int(args.siminst if args.siminst is not None else 0)
OUTDIR                       = args.outdir if args.outdir is not None else 0
RESSURECT                    = args.resurrect
INDTYPE                      = args.type
POPN                         = args.initial_number if args.initial_number is not None else 1000
SHIFTING_FITNESS_PEAK        = args.shifting_peak
LS_INCREMENT                 = float(args.landscape_increment)
MUTATION_RATE_ALLELE         = 5 if args.almrate is None else float(args.almrate)  # ? in entry-mutations per generation
MUTATION_RATE_CONTRIB_CHANGE = 5 if args.gpmrate is None else float(args.gpmrate)  # ? in mutations per generation

LS_SHIFT_EVERY   = int(1e4)
STD              = 1
AMPLITUDE        = 1
LOG_EVERY        = GENERATION*10
PICKING_MEAN_STD = (0, 0.5) #recombination
BEGIN_DATE       = datetime.now().strftime("%I:%M%p on %B %d, %Y")


INDIVIDUAL_INITS = {
    #    One-to-One
    "1": {
        'trait_n': 4,
        'alleles': np.array([1, 1, 1, 1], dtype=np.float16),
        'coefficients': np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
    },
    #    modular
    "2": {
        'trait_n': 4,
        'alleles': np.array([1, 1, 1, 1], dtype=np.float16),
        'coefficients': np.array([
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ], dtype=np.float16)
    },
    #    All Connected
    "3": {

        'trait_n': 4,
        'alleles': np.array([1, 1, 1, 1], dtype=np.float16),
        'coefficients': np.array([
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ], dtype=np.float16)
    },
}

# Creates directories with variance and covariance data + fitness data
#[os.makedirs(os.path.join(OUTDIR, intern_path), exist_ok=True) for intern_path in ['var_covar', 'fitness_data']]

# Creating mutation plan for an entire period of time (for eg: an entire generation or the entire run)
# Pick a number from poisson

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

# Actual mutation happening for contribution
def mutate_gpmap(contributions):
    probs = np.random.uniform(low=0, high=1, size=(4, 4)).round(4)
    rows, cols = probs.shape

    for i in range(rows):
        for j in range(cols):
            if probs[i, j] <= MUTATION_RATE_CONTRIB_CHANGE:
                pick = np.random.normal(*PICKING_MEAN_STD, 1)
                contributions[i, j] += pick

# Actual mutation happening for alleles
def mutate_alleles(alleles: np.ndarray) -> None:
    for g in range(alleles.shape[0]):
        if np.random.uniform() <= MUTATION_RATE_ALLELE:
            pick = np.random.normal(*PICKING_MEAN_STD, 1)
            alleles[g] += pick

class FitnessMap:
    std = 1

    @classmethod
    def getmap(cls, mean):
        u = mean
        exp = math.exp

        def _(phenotype: np.ndarray):
            return AMPLITUDE * exp(-(np.sum(((phenotype - u) ** 2) / (2 * cls.std ** 2))))

        return _

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
        self.mutation_plan_contrib = make_mutation_plan_contrib(MUTATION_RATE_CONTRIB_CHANGE)
        self.mutation_plan_alleles = make_mutation_plan_alleles(MUTATION_RATE_ALLELE)

        # ? ------------------------------ [ AGtrs ]
        # What I am saving into the file
        self.agg = {
              "began_logging_at"     : -1,
              "logged_every"         : LOG_EVERY,
              "weight_vectors"       : [],
              "contribution_matrices": [],
              "optima_positions"     : [],
             "elapsed"              : 0
        }
        # self.fitmean_agg = np.array([])

    #     state = {
    #         "last_iteration": self.it,
    #         "alleles": self.ALLS,
    #         "gpms": self.GPMS,
    #         "fitness_mean": self.mean
    #     }
    #
    #     with open(os.path.join(OUTDIR, 'state_{}.pkl'.format(REPLICATE_N)), 'wb') as f: pickle.dump(state, f)
    #Function related to Brith
    def pick_parent(self) -> int:

        indices = np.arange(len(self.PHNS))
        fitnesses = np.array([*map(FitnessMap.getmap(self.mean), self.PHNS)])
        cumfit = reduce(lambda x, y: x + y, fitnesses)

        return np.random.choice(indices, p=fitnesses / cumfit)

    def save_aggregate(self)->None:
        outpath = os.path.join(OUTDIR, "popinfo_replicate{}.json".format(REPLICATE_N))
        with open(outpath, 'w') as outfile:
            _ =          {
              "began_logging_at"     : self.agg['began_logging_at'],
              "logged_every"         : LOG_EVERY,
              "weight_vectors"       : np.array( self.agg['weight_vectors'] ).tolist(),
              "contribution_matrices": np.array(self.agg['contribution_matrices']).tolist(),
              "optima_positions"     : np.array(self.agg['optima_positions']).tolist(),
              "elapsed"              : 0
            }

            json.dump(_, outfile)
            print("Wrote to {}.".format(outpath))
    #Function related to Death
    def pick_death(self) -> int:
        indices = np.arange(len(self.PHNS))
        return np.random.choice(indices)
    #Birth-death process: what happens in a iteration.
    def tick(self):
        if self.it > ITEND - (math.ceil((ITEND - ITSTART) / 10)) and not (self.it % (LOG_EVERY)):
            if self.agg['began_logging_at']<0: # So it counts from the iteration we are at
                self.agg['began_logging_at']       = self.it


            self.agg["weight_vectors"]        = self.ALLS if len( self.agg["weight_vectors"] ) == 0 else np.vstack(  (self.agg["weight_vectors"],self.ALLS ) )
            self.agg["contribution_matrices"] = self.GPMS if len( self.agg["contribution_matrices"] ) == 0 else np.vstack( ( self.agg["contribution_matrices"],self.GPMS) )
            self.agg['optima_positions']      = self.mean if len(self.agg['optima_positions']) == 0 else np.vstack((self.agg['optima_positions'], self.mean))
            self.agg['elapsed']               += LOG_EVERY # How many iterations passed

        if (not self.it % LS_SHIFT_EVERY):
            self.shift_landscape(LS_INCREMENT, SHIFTING_FITNESS_PEAK)

        death_index = self.pick_death()
        birth_index = self.pick_parent()

        _alleles = np.copy(self.ALLS[birth_index])
        _contribs = np.copy(self.GPMS[birth_index])
        # Checking if according to mutation plan whether there is a mutation happening at this birth
        while bool(len(self.mutation_plan_alleles['iterns'])) and self.it % GENERATION == \
                self.mutation_plan_alleles['iterns'][0]:
            posn = self.mutation_plan_alleles['posns'][0]
            _alleles[posn] += np.random.normal(*PICKING_MEAN_STD)
            self.mutation_plan_alleles['iterns'] = self.mutation_plan_alleles['iterns'][1:]
            self.mutation_plan_alleles['posns'] = self.mutation_plan_alleles['posns'][1:]

        while bool(len(self.mutation_plan_contrib['iterns'])) and self.it % GENERATION == \
                self.mutation_plan_contrib['iterns'][0]:
            posn = tuple(self.mutation_plan_contrib['posns'][0])
            _contribs[posn] += np.random.normal(*PICKING_MEAN_STD)
            self.mutation_plan_contrib['iterns'] = self.mutation_plan_contrib['iterns'][1:]
            self.mutation_plan_contrib['posns'] = self.mutation_plan_contrib['posns'][1:]

        if self.it % GENERATION == 0:
            self.mutation_plan_contrib = make_mutation_plan_contrib(MUTATION_RATE_CONTRIB_CHANGE)
            self.mutation_plan_alleles = make_mutation_plan_alleles(MUTATION_RATE_ALLELE)

        self.PHNS[death_index] = _contribs @ _alleles.T
        self.ALLS[death_index] = _alleles
        self.GPMS[death_index] = _contribs
        self.it += 1

    def get_avg_fitness(self) -> float:
        return reduce(lambda x, y: x + y, map(FitnessMap.getmap(self.mean), self.PHNS)) / len(self.PHNS)

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
            # *Large IT
            if abs(LANDSCAPE_INCREMENT) > 0.9:
                if np.max(self.mean) > 0.9:
                    if np.random.choice([-1, 1]) > 0:
                        self.mean -= LANDSCAPE_INCREMENT
                elif np.min(self.mean) < -0.9:
                    if np.random.choice([-1, 1]) > 0:
                        self.mean += LANDSCAPE_INCREMENT
                else:
                    self.mean += np.random.choice([-1, 1])
                # *Small IT
            else:
                if np.max(self.mean) > 0.9:
                    if np.random.choice([1, -1]) > 0:
                        self.mean -= LANDSCAPE_INCREMENT
                    else:
                        ...
                elif np.min(self.mean) < -0.9:
                    if np.random.choice([1, -1]) > 0:
                        self.mean += LANDSCAPE_INCREMENT
                    else:
                        ...
                else:
                    self.mean += np.random.choice([1, -1]) * LANDSCAPE_INCREMENT
        # ? Uncorrelated shifts
        elif CORRELATED == 'uncorrelated':
            # *Large IT
            if abs(LANDSCAPE_INCREMENT) > 0.9:
                for i, x in enumerate(self.mean):
                    if self.mean[i] > 0.9:
                        if np.random.choice([1, -1]) > 0:
                            self.mean[i] -= LANDSCAPE_INCREMENT
                        else:
                            ...
                    elif self.mean[i] < -0.9:
                        if np.random.choice([1, -1]) > 0:
                            self.mean[i] += LANDSCAPE_INCREMENT
                        else:
                            ...

                    else:
                        self.mean[i] += np.random.choice([1, -1]) * LANDSCAPE_INCREMENT
            # *Small IT
            else:
                for i, x in enumerate(self.mean):
                    if self.mean[i] > 0.9:
                        if np.random.choice([1, -1]) > 0:
                            self.mean[i] -= abs(LANDSCAPE_INCREMENT)
                        else:
                            ...
                    elif self.mean[i] < -0.9:
                        if np.random.choice([1, -1]) > 0:
                            self.mean += LANDSCAPE_INCREMENT
                        else:
                            ...
                    else:
                        coin = np.random.choice([1, -1])
                        self.mean[i] += coin * LANDSCAPE_INCREMENT

        elif CORRELATED == 'pairwise':
            # *Large IT
            if abs(LANDSCAPE_INCREMENT) > 0.9:
                for pair_inds in [[0, 1], [2, 3]]:
                    if self.mean[pair_inds][0] > 0.9:
                        if np.random.choice([1, -1]) > 0:
                            self.mean[pair_inds] -= LANDSCAPE_INCREMENT
                        else:
                            ...
                    elif self.mean[pair_inds][0] < -0.9:
                        if np.random.choice([1, -1]) > 0:
                            self.mean[pair_inds] += LANDSCAPE_INCREMENT
                        else:
                            ...

                    else:
                        self.mean[pair_inds] += np.random.choice([1, -1]) * LANDSCAPE_INCREMENT
            # *Small IT
            else:
                for pair_inds in [[0, 1], [2, 3]]:
                    if self.mean[pair_inds][0] > 0.9:
                        if np.random.choice([1, -1]) > 0:
                            self.mean[pair_inds] -= abs(LANDSCAPE_INCREMENT)
                        else:
                            ...
                    elif self.mean[pair_inds][0] < -0.9:
                        if np.random.choice([1, -1]) > 0:
                            self.mean += LANDSCAPE_INCREMENT
                        else:
                            ...
                    else:
                        coin = np.random.choice([1, -1])
                        self.mean[pair_inds] += coin * LANDSCAPE_INCREMENT


        elif CORRELATED == '8_bimodular':

            # *Large IT
            if abs(LANDSCAPE_INCREMENT) > 0.9:
                for pair_inds in [[0, 1, 2 ,3], [4,5,6,7]]:
                    if self.mean[pair_inds][0] > 0.9:
                        if np.random.choice([1, -1]) > 0:
                            self.mean[pair_inds] -= LANDSCAPE_INCREMENT
                        else:
                            ...
                    elif self.mean[pair_inds][0] < -0.9:
                        if np.random.choice([1, -1]) > 0:
                            self.mean[pair_inds] += LANDSCAPE_INCREMENT
                        else:
                            ...
                    else:
                        self.mean[pair_inds] += np.random.choice([1, -1]) * LANDSCAPE_INCREMENT
            # *Small IT
            else:
                for pair_inds in [[0, 1, 2 ,3], [4,5,6,7]]:

                    if self.mean[pair_inds][0] > 0.9:
                        if np.random.choice([1, -1]) > 0:
                            self.mean[pair_inds] -= abs(LANDSCAPE_INCREMENT)
                        else:
                            ...

                    elif self.mean[pair_inds][0] < -0.9:
                        if np.random.choice([1, -1]) > 0:
                            self.mean += LANDSCAPE_INCREMENT
                        else:
                            ...
                    else:
                        coin = np.random.choice([1, -1])
                        self.mean[pair_inds] += coin * LANDSCAPE_INCREMENT
        elif CORRELATED == '8_quadmodular':

            # *Large IT
            if abs(LANDSCAPE_INCREMENT) > 0.9:
                for pair_inds in [[0, 1], [2, 3],[4,5],[6,7]]:
                    if self.mean[pair_inds][0] > 0.9:
                        if np.random.choice([1, -1]) > 0:
                            self.mean[pair_inds] -= LANDSCAPE_INCREMENT
                        else:
                            ...
                    elif self.mean[pair_inds][0] < -0.9:
                        if np.random.choice([1, -1]) > 0:
                            self.mean[pair_inds] += LANDSCAPE_INCREMENT
                        else:
                            ...

                    else:
                        self.mean[pair_inds] += np.random.choice([1, -1]) * LANDSCAPE_INCREMENT
            # *Small IT
            else:
                for pair_inds in [[0, 1], [2, 3],[4,5],[6,7]]:
                    if self.mean[pair_inds][0] > 0.9:
                        if np.random.choice([1, -1]) > 0:
                            self.mean[pair_inds] -= abs(LANDSCAPE_INCREMENT)
                        else:
                            ...
                    elif self.mean[pair_inds][0] < -0.9:
                        if np.random.choice([1, -1]) > 0:
                            self.mean += LANDSCAPE_INCREMENT
                        else:
                            ...
                    else:
                        coin = np.random.choice([1, -1])
                        self.mean[pair_inds] += coin * LANDSCAPE_INCREMENT


        else:
            exit("Unspecified landscape shift type")

def print_receipt() -> None:
    receipt = {
        "ITSTART": ITSTART,
        "ITEND": ITEND,
        "REPLICATE_N": REPLICATE_N,
        "OUTDIR": OUTDIR,
        "RESSURECTED": RESSURECT,
        "INDTYPE": INDTYPE,
        "POPN": POPN,
        "SHIFTING_FITNESS_PEAK": SHIFTING_FITNESS_PEAK,
        "LS_INCREMENT": LS_INCREMENT,
        "MUTATION_RATE_ALLELE": MUTATION_RATE_ALLELE,
        "MUTATION_RATE_CONTRIB_CHANGE": MUTATION_RATE_CONTRIB_CHANGE,
        "LS_SHIFT_EVERY": LS_SHIFT_EVERY,
        "STD": STD,
        "AMPLITUDE": AMPLITUDE,
        "LOG_EVERY": LOG_EVERY,
        "date_finished": datetime.now().strftime("%I:%M%p on %B %d, %Y"),
        "date_started": BEGIN_DATE,
        "PICKING_MEAN_STD": [*PICKING_MEAN_STD]
    }
    with open(os.path.join(OUTDIR, "parameters_replicate{}.json".format(REPLICATE_N)), 'w') as infile:
        json.dump(receipt, infile)

# This is not updaated
def ressurect():
    state_loc = os.path.join(OUTDIR, 'state_{}.pkl'.format(REPLICATE_N))

    with open(state_loc, 'rb') as inf:
        state = pickle.load(inf)
        it = state['last_iteration'] #we don't have this information as it is.
        ITSTART = it
        alls = state['alleles'] #this needs to become weight_vector
        gpms = state['gpms'] #this needs to become contribution_matrix
        phns = np.array([gpms[i] @ alls[i].T for i in range(alls.shape[0])], dtype=np.float16)
        # fitmap  = FitnessMap       (1)
        fitmean = state['fitness_mean']

        if ITEND <= it:
            print(f"End iteration that was specified {ITEND} is lower than this population's 'age'({it}). Exited ")
            exit(1)

        return [Universe(it,
                         alls,
                         gpms,
                         phns,
                         fitmean)
            ,
                ITSTART]

if RESSURECT:
    [u, start_iter] = ressurect()
    ITSTART = start_iter
    for it in range(ITSTART, ITEND + 1): u.tick()
else:
    alls = np.array([INDIVIDUAL_INITS[str(INDTYPE)]['alleles'] for i in range(POPN)], dtype=np.float16)
    gpms = np.array([INDIVIDUAL_INITS[str(INDTYPE)]['coefficients'] for i in range(POPN)], dtype=np.float16)
    phns = np.array([gpms[i] @ alls[i].T for i in range(POPN)], dtype=np.float16)
    fitmean = np.array([0, 0, 0, 0], dtype=np.float16)

    u = Universe(ITSTART,
                 alls,
                 gpms,
                 phns,
                 fitmean)
    for it in range(ITSTART, ITEND + 1): u.tick()

if OUTDIR:
    u.save_aggregate()
    print_receipt()
