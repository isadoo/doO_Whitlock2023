from dataclasses import replace
from pprint import pprint
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
parser = argparse.ArgumentParser (                                          description           =                             'Simulation presets'                                                                                                                                          )
parser .add_argument ("-it"       , "--itern"               , type= int      ,                 help = "The number of iterations"                                                                                            )
parser .add_argument ('-save'     , '--outdir'                  , type   = dir_path     ,                                help = ""                                                                                           "Specify the path to write the results of the simulation.""" )
parser .add_argument ("-itstart"  , "--iter_start"              , type   = int          ,required =True,                 help = "The number of iterations"                                                                                                                                )
parser .add_argument ("-itend"    , "--iter_end"                , type   = int          ,required =True,                 help = "The number of iterations"                                                                                                                                )
parser .add_argument ("-ls"       , "--landscape_increment"     , type   = float        ,required =True,                 help = "Simulation tag for the current instance."                                                                                                                )
parser .add_argument ("-sim"      , "--siminst"                 , type   = int          ,                                help = "Simulation tag for the current instance."                                                                                                                )
parser .add_argument ("-SP"       , "--shifting_peak"           , type   = str         ,
                    choices =['correlated', 'uncorrelated', 'pairwise', '8_bimodular','8_quadmodular'],
                    required=True, help = "Flag for whether the fitness landscape changes or not.")
parser .add_argument ('-t'        , '--type'                    , type   = int          ,required =True,                 help = 'Types involved in experiment'                                                                                                                            )
parser .add_argument ('-initn'    , '--initial_number'          , type   = int          ,                                help = 'Starting number of individuals'                                                                                                                          )
parser .add_argument ('-gpm_rate' , '--gpmrate'                 , type   = float        ,                                help = 'GP-map contribution change mutation rate'                                                                                                                )
parser .add_argument ('-alm_rate' , '--almrate'                 , type   = float        ,                                help = 'Allelic mutation rate'                                                                                                                                   )
parser .add_argument ('-re'       , '--resurrect'               , type   = dir_path     ,                                help = 'Path to reinstate the population from.'                                                                                                                  )

# Parameters for simulation
args                         = parser .parse_args()
GENERATION 					 = 1000
ITSTART                      = int(args.iter_start)
ITEND                        = int(args.iter_end)
REPLICATE_N                  = int (args .siminst if args.siminst is not None else 0)
OUTDIR                       = args.outdir if args.outdir is not None else 0
RESSURECT_PATH               = args.resurrect if args.resurrect is not None else 0
INDTYPE                      = args.type
POPN                         = args.initial_number if args.initial_number is not None else 1000
SHIFTING_FITNESS_PEAK        = args.shifting_peak
LS_INCREMENT                 = float(args.landscape_increment)

MUTATION_RATE_ALLELE         = 5 if args.almrate is None else float(args.almrate ) #? in entry-mutations per generation
MUTATION_RATE_CONTRIB_CHANGE = 5 if args.gpmrate is None else float( args.gpmrate ) #? in mutations per generation

LS_SHIFT_EVERY   = int(1e4)
STD              = 1
AMPLITUDE        = 1
LOG_EVERY        = GENERATION*10
PICKING_MEAN_STD = (0, 0.5) #mutation
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
    "8": {
        'trait_n': 8,
        'alleles': np.array([1, 1, 1, 1,1,1,1,1], dtype=np.float16),
        'coefficients': np.array([
		[1,0,0,0,0,0,0,0],
		[0,1,0,0,0,0,0,0],
		[0,0,1,0,0,0,0,0],
		[0,0,0,1,0,0,0,0],
		[0,0,0,0,1,0,0,0],
		[0,0,0,0,0,1,0,0],
		[0,0,0,0,0,0,1,0],
		[0,0,0,0,0,0,0,1],
        ], dtype=np.float16)
    },
}

#[ os.makedirs(os.path.join(OUTDIR, intern_path), exist_ok=True) for intern_path in ['var_covar','fitness_data']]

def make_mutation_plan_alleles(
	_lambda: float,
	trait_n: int,
	period : int=GENERATION,
	)      : 

	"""
	@lambda - the rate of the poisson distribution, in this case -- mutate per generation
	@period - number the interval over which to pick, in this case a single generation
	"""

	#? how many mutations occur in a given period (Generation)
	poolsize = np.random.poisson(_lambda)

	#? at which iterations do they occur
	iterns = np.random.randint(low=0, high=1000, size=poolsize); iterns.sort()

	#? at which positions do they occur?

	entries = random.sample(range (1,(period*trait_n+1)),poolsize); entries.sort();
	posns   = np.array([*map (lambda x: x%trait_n, entries)])

	return {
		"posns" : posns,
		"iterns": iterns
	}

def make_mutation_plan_contrib(
	_lambda:float,
	trait_n:int,
	period :int=GENERATION):
	"""
	@lambda - the rate of the poisson distribution, in this case -- mutrate per generation
	@period - number the interval over which to pick, in this case a single generation
	"""
	#? how many mutations occur in a given period (Generation)
	poolsize   =   np.random.poisson(_lambda)

	#? at which iterations do they occur
	iterns = np.random.randint(low=0, high=1000, size=poolsize); iterns.sort()

	#? at which positions do they occur?
	entries = random.sample(range (1,( period*trait_n +1 )),poolsize);
	entries.sort();
	posns   = np.array([*map (lambda x: ((x%(trait_n**2))//trait_n,(x%(trait_n**2))%trait_n), entries)])
	return {
		"posns" : posns,
		"iterns": iterns
	}

def mutate_gpmap(contributions,trait_n:int):


	print("Contriubtions shape:",contributions.shape())
	probs = np.random.uniform(low=0, high=1, size=(trait_n,trait_n)).round(4)

	rows, cols = probs .shape

	for i in  range(rows):
		for j in range(cols):
			if probs[i,j] <= MUTATION_RATE_CONTRIB_CHANGE:
				pick = np.random.normal(*PICKING_MEAN_STD, 1)
				contributions[i,j] += pick

def mutate_alleles(alleles:np.ndarray):
	for g in range(alleles.shape[0]):
		if np.random.uniform() <= MUTATION_RATE_ALLELE:
			pick = np.random.normal(*PICKING_MEAN_STD, 1)
			alleles[g] += pick

class FitnessMap:

	def __init__(self, std):
		self.std = std
	def getmap(self, mean):

		u   = mean
		exp = math.exp

		def _(phenotype:np.ndarray):
			return AMPLITUDE * exp(-(np.sum(((phenotype - u)**2)/(2*self.std**2))))
		return _

# everything together
class Universe:

	def __init__(
		self,
		current_iter:int,
		trait_n: int,
		ALLS: np.ndarray,
		GPMS: np.ndarray,
		PHNS: np.ndarray,
		fmap: FitnessMap,
		mean: np.ndarray,
		) -> None:


		# ? ------------------------------ [ STATE ]
		self.it      = current_iter
		self.trait_n = trait_n
		self.ALLS    = ALLS
		self.GPMS    = GPMS
		self.PHNS    = PHNS

		# ? ------------------------------ [ ENV ]
		self.fitmap                = fmap
		self.mean                  = mean
		self.mutation_plan_contrib = make_mutation_plan_contrib(MUTATION_RATE_CONTRIB_CHANGE,self.trait_n)
		self.mutation_plan_alleles = make_mutation_plan_alleles(MUTATION_RATE_ALLELE,self.trait_n)

		# What I am saving into the file
		self.agg = {
			  "began_logging_at"      : -1,
			  "logged_every"         : LOG_EVERY,
			# 'covar_slices'         : np.array([np.cov(self.PHNS.T)]),
			  "weight_vectors"       : [],
			  "contribution_matrices": [],
			  "optima_positions"      : [],
			  "elapsed"              : 0
		}
		#self.fitmean_agg   = np.array([])

	def save_aggregate(self) -> None:
		outpath =os.path.join(OUTDIR, "popinfo_replicate{}.json".format(REPLICATE_N))
		_ =          {
			"began_logging_at"     : self.agg['began_logging_at'],
			"logged_every"         : LOG_EVERY,
			"weight_vectors"       : self.agg['weight_vectors'].tolist(),
			"contribution_matrices": self.agg['contribution_matrices'].tolist(),
			"optima_positions"     : self.agg['optima_positions'].tolist(),
			"elapsed"              : 0
		}

		with open(outpath,'w') as outfile:
			json.dump(_, outfile)
			print("Wrote to {}.".format(outpath))

	def pick_parents(self)->List[int]:
		indices   = np.arange(len( self.PHNS ))
		fitnesses = np.array( [  *map( Fitmap.getmap(self.mean), self.PHNS)] )
		cumfit    = reduce(lambda x,y : x+y, fitnesses)
		return np.random.choice(indices,2,p=fitnesses/cumfit, replace=False)

	def pick_death(self)->int:
		indices   = np.arange(len( self.PHNS ))
		return np.random.choice(indices)

	def birth_death(self):

		""""A birth now constitutes recombination of two individuals alleles, genes.
		random vector of [0,1]  of len 20 to define positions. 
		Creat alleles, contributions.
		Apply mutations.
		Change in place.
		"""


		death_index = self.pick_death()
		[ xy,xx ]   = self.pick_parents()


		mask        = np.random.choice([0,1], size=(self.trait_n**2+self.trait_n,)).reshape((self.trait_n+1,self.trait_n))

		alls_xx     = np.copy(self.ALLS[xx])
		contribs_xx = np.copy(self.GPMS[xx])

		alls_xx    [mask[:1][0]==1] = self.ALLS[xy][mask[:1][0]==1]
		contribs_xx[mask[1:]   ==1] = self.GPMS[xy][mask[1: ]  ==1]

		while bool(len(self.mutation_plan_alleles['iterns'])) and self.it % GENERATION == self.mutation_plan_alleles['iterns'][0]:
			posn = self.mutation_plan_alleles['posns'][0]

			alls_xx[posn] += np.random.normal(*PICKING_MEAN_STD)

			self.mutation_plan_alleles['iterns'] = self.mutation_plan_alleles['iterns'][1:]
			self.mutation_plan_alleles['posns' ] = self.mutation_plan_alleles['posns' ][1:]

		while bool(len(self.mutation_plan_contrib['iterns'])) and self.it % GENERATION == self.mutation_plan_contrib['iterns'][0]:
			posn = tuple(self.mutation_plan_contrib['posns'][0])
			contribs_xx[posn] += np.random.normal(*PICKING_MEAN_STD)
			self.mutation_plan_contrib['iterns'] = self.mutation_plan_contrib['iterns'][1:]
			self.mutation_plan_contrib['posns' ] = self.mutation_plan_contrib['posns' ][1:]
			
		if self.it % GENERATION == 0:

			self.mutation_plan_contrib = make_mutation_plan_contrib(MUTATION_RATE_CONTRIB_CHANGE,self.trait_n)
			self.mutation_plan_alleles = make_mutation_plan_alleles(MUTATION_RATE_ALLELE,self.trait_n)

		self.PHNS[death_index] = contribs_xx @ alls_xx.T
		self.ALLS[death_index] = alls_xx
		self.GPMS[death_index] = contribs_xx

	def tick(self):
		if self.it > ITEND - (math.ceil((ITEND - ITSTART) / 10)) and not (self.it % (LOG_EVERY)):
			if self.agg['began_logging_at']<0: # So it counts from the iteration we are at
				self.agg['began_logging_at']       = self.it


			self.agg["weight_vectors"]        = self.ALLS if len( self.agg["weight_vectors"] ) == 0 else np.vstack(  (self.agg["weight_vectors"],self.ALLS ) )
			self.agg["contribution_matrices"] = self.GPMS if len( self.agg["contribution_matrices"] ) == 0 else np.vstack( ( self.agg["contribution_matrices"],self.GPMS) )
			self.agg['optima_positions']      = self.mean if len(self.agg['optima_positions']) == 0 else np.vstack((self.agg['optima_positions'], self.mean))
			self.agg['elapsed']               += LOG_EVERY # How many iterations passed


		if ( not self.it % LS_SHIFT_EVERY ):
			self.shift_landscape(LS_INCREMENT,SHIFTING_FITNESS_PEAK)

		self.birth_death()

		self.it += 1

	def get_avg_fitness(self)->float:
		return reduce(lambda x,y: x + y, map(self.fitmap.getmap(self.mean), self.PHNS))/len(self.PHNS)

	#def write_covar_pkl(self,outdir):

	#	outfile    = os.path.join(outdir,'var_covar','mean_var_covar_{}.pkl'.format(REPLICATE_N))
	#	_ = {
	#		'covar_slices'   : self.covar_agg['covar_slices'],
	#		'elapsed'        : self.covar_agg[ 'elapsed'         ],
	#		'began_loggin_at': self.covar_agg[ 'began_loggin_at' ],
	#		'logged_every'   : self.covar_agg[ 'logged_every'    ]
	#	}
	#	with open(outfile,'wb') as log:
	#		pickle.dump(_, log)

	#def log_var_covar(self):

	#	self.covar_agg['covar_slices'] = np.row_stack([ self.covar_agg['covar_slices'],np.array([ np.cov(self.PHNS.T) ] ) ] )
	#	self.covar_agg['elapsed']                              += 1
	#	if self.covar_agg['began_loggin_at'] == -1:
	#		self.covar_agg['began_loggin_at']= self.it

	def shift_landscape(
			self,
			LANDSCAPE_INCREMENT: float,
			CORRELATED: bool) -> None:

		# ? Correlated shifts
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

def print_receipt()->None:
	receipt = {
		  "ITSTART"                      : ITSTART                                                            ,
		  "ITEND"                        : ITEND                                                              ,
		  "REPLICATE_N"                   : REPLICATE_N                                                         ,
		  "OUTDIR"                       : OUTDIR                                                             ,
		  "RESSURECT_PATH"               : RESSURECT_PATH                                                     ,
		  "INDTYPE"                      : INDTYPE                                                            ,
		  "POPN"                         : POPN                                                               ,
		  "SHIFTING_FITNESS_PEAK"        : SHIFTING_FITNESS_PEAK                                              ,
		  "LS_INCREMENT"                 : LS_INCREMENT                                                       ,
		  "MUTATION_RATE_ALLELE"         : MUTATION_RATE_ALLELE                                               ,
		  "MUTATION_RATE_CONTRIB_CHANGE" : MUTATION_RATE_CONTRIB_CHANGE                                       ,
		  "LS_SHIFT_EVERY"                : LS_SHIFT_EVERY                                                      ,
		  "STD"                          : STD                                                                ,
		  "AMPLITUDE"                    : AMPLITUDE                                                          ,
		  "LOG_EVERY"                : LOG_EVERY                                                      ,
		  "date_finished"                : datetime                    .now().strftime("%I:%M%p on %B %d, %Y"),
		  "date_started"                 : BEGIN_DATE                                                         ,
		"PICKING_MEAN_STD" : [*PICKING_MEAN_STD]
	}
	with open(os.path.join(OUTDIR, "parameters_replicate{}.json".format(REPLICATE_N)),'w') as infile:
		json.dump(receipt, infile)

#***************** INITS ***************

TRAIT_N = INDIVIDUAL_INITS[str( INDTYPE )]['alleles' ].shape[0]
alls    = np.array([ INDIVIDUAL_INITS[str( INDTYPE )]['alleles' ] for i in range(POPN) ], dtype=np.float64)
gpms    = np.array([ INDIVIDUAL_INITS[str( INDTYPE )]['coefficients'] for i in range(POPN)	], dtype=np.float64)
phns    = np.array( [gpms[i]@ alls[i].T for i in range(POPN) ], dtype=np.float64)

Fitmap   = FitnessMap      (1)

universe = Universe        (ITSTART,TRAIT_N,
									alls,
									gpms,
									phns,
									Fitmap,
									np.array([0]*TRAIT_N, dtype=np.float64))


for it in range(ITSTART, ITEND+1): universe.tick()

if OUTDIR:
	universe.save_aggregate()
	print_receipt()

