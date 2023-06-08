# doO_Whitlock2023
*Scripts related to analyses and simulations for paper with Mike Whitlock 2023*

Here are the detailed instructions on how to rerun all these scripts from creating the data to analyzing it. 
What is there in each folder: 

**Simulation Scripts**

- `evo_covar.py` (Figure 1,2 and 3): this python script simulates asexual populations in correlated or independent (same thing as uncorrelated), or block correlated (same as pairwise) environments. You can decide how often the environment changes and the mutation rates along with a few other parameters.

- `evo_recomb.py` (Figure1) : same as previous but for sexual populations!

- `evo_covar8x8.py` (Figure supmat) : this one allows to make larger C matrices, specifically 8 by 8

- `UNBOUNDED_evolvability_simulation.py` (Figure 3): here there are no mutations on C (but you can reactivate them by uncommenting that portion of the code) and there are no bounds on how far the environment can change (line 260). 

**Analysis Scripts:**

- `LD_v_Pleiotropy.py` : (Figure 1) Calculates the G matrix for two given experiments. It also separates G into Total G, Pleiotropy portion of G, LD portion of G. This script returns csvs separating the elements of G in columns, and rows represent replicate populations. It also prints out the confidence intervals and means for each G matrix measured. This script takes two experiments because it also compares the results of the two experiments. It was written to be used on the sexual and asexual populations on the environments correlated and independent.  (Figure is generated from this output)

- `SEX_ldVpleiotropy.py` : (Figure 1)  It's exactly the same thing as the one above, but the csv files that it produces have the tag “SEX”.  (Figure is generated from this output)

- `BLOCKS_ldVpleiotropy.py` : (Figure 2) Very similar to the ones before. The main difference is that this one was written for also comparing the G matrices of populations evolved in block correlated environments. It has a part that compares within and between blocks.  (Figure is generated from this output)

- `AdditiveVar.py`: (Figure 1) Calculates the additive variance and covariance of sexual populations. (Figure is generated from this output)

- `fitness_consequences.py` : (Figure 3) It takes resurrected populations that have been re-exposed to one of the three environments and outputs the fitness of each timepoint. The output file is a csv of fitness per time point. The input file should be a pickled file. (Figure is generated from this output)
- `MutVar.py`: (Figure ?) It takes csvs of two given experiments, samples 1 million individuals from each replicate population. It calculates the “a” vector portion of the mutational variance-covariance. The output is a csv with the mutational variances and covariances of each trait and pair of traits, for the two experiments compared.
- `MutVar_c.py`: Same as above but for  the “c” matrix portion of the mutation variance-covariance. (There are a few differences in the functions used here)
- `8x8__BLOCKS_ldVpleiotropy.py`: same function of `BLOCKS_ldVpleiotropy.py`  but used only in supplementary figures. 


**Treating Data Scripts**

- `json_to_csv.py` : (All figures)  All simulation outputs are json files, this script converts them to csv.
- `GrabLastIt_.py`: (Figure 3) Grabs last iteration of a simulation and prepares it to run again 
- `evolvability_C.py` : (Figure 3) From the last iteration it will find the most common c matrix and force it into all individuals. (This is the script used to create: exp111-allsameC.pickle, exp112-allsameC.pickle,exp311-allsameC.pickle)

**Steps for each figure:**

Figure 1 Left side (Asexual), steps: 
1- run `evo_covar.py`
2 - run `json_to_csv.py`, 
3 - to erase all unnecessary parenthesis on the terminal you can do `sed -i -e ‘s/\]//g’ -e ‘s/\[//g’ *.csv `
4- run `LD_v_Pleiotropy.py`.
Figure 1 Right side (Sexual): 
1- run `evo_covar.py`
2 - run `json_to_csv.py`, 
3 - to erase all unnecessary parenthesis on the terminal you can do `sed -i -e ‘s/\]//g’ -e ‘s/\[//g’ *.csv`
4 - run `SEX_ldVpleiotropy.py`
5 - run `AdditiveVar.py`

Figure 2: 
1- run `evo_covar.py`
2 - run `json_to_csv.py`, 
3 - to erase all unnecessary parenthesis on the terminal you can do `sed -i -e ‘s/\]//g’ -e ‘s/\[//g’ *.csv`
4 - run `BLOCKS_ldVpleiotropy.py`.

Figure 3 is the transplant-like experiment. 
First you would have run the simulations using `evo_covar.py`. 
Second, once simulations are done running you use `GrabLastIt_.py` to get the last iteration
Third, you use `evolvability_C.py` to make all C the same 
Fourth, you rerun each experiment on one of the environments using UNBOUNDED_evolvability_simulation.py. This script does each environment separately, so to expose them to all three, you have to run it three times changing the “shifting_peaks” argument you pass when running the python script. 

Simulation Arguments:
To run our simulation scripts you will have to pass certain arguments. 
Here is an example of the minimum you would need to specify:
`python3 evo_recomb.py -t 2 -itstart 0 -itend 5000000 -SP uncorrelated --outdir $path/to/your/output/directory -ls 0.2`
The explanation for each argument is present in the beginning of each simulation script. 
