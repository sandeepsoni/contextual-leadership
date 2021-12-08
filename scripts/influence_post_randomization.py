import argparse
import random
import pickle
import numpy as np
import pandas as pd
import os
import sys
if not os.path.abspath ("../modules") in sys.path:
	sys.path.append (os.path.abspath ("../modules"))
import hpmodels 

import logging
logging.basicConfig(
	level=logging.INFO, 
	format= '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
	datefmt='%H:%M:%S'
)

def readArgs ():
	parser = argparse.ArgumentParser (description="Bandwidth grid search")
	parser.add_argument ("--data-file", type=str, required=True, help="The name of the file containing all the data")
	parser.add_argument ("--log-file", type=str, required=True, help="The name of the file containing all the logs")
	parser.add_argument ("--params-file", type=str, required=True, help="The name of the file containing all the params")
	parser.add_argument ("--bandwidth", type=float, required=False, default=2.0, help="Bandwidth parameter")
	parser.add_argument ("--ncascades", type=int, required=False, default=None, help="The number of cascades to consider")
	parser.add_argument ("--l2-coeff", type=float, required=False, default=0.0, help="L2 coeffient penalty")
	parser.add_argument ("--seed", type=int, required=False, default=42, help="random seed")
	args = parser.parse_args ()
	return args

def randomize_cascades (cascades, seed):
	# serialize
	serialized_cascades = [(i,src,timestamp) for i, cascade in enumerate (cascades) for timestamp in sorted (cascade) for src in cascade[timestamp]]
	df = pd.DataFrame (serialized_cascades, columns=["cascade_number", "source", "timestamp"])

	# shuffle
	np.random.seed (seed)
	#df["source"] = np.random.permutation (df["source"].values) # simple randomization
	df['source'] = df.groupby('timestamp', as_index=False).source.transform(np.random.permutation) #randomize within a timestamp

	new_cascades = [{} for cascade in cascades]
	# deserialize
	for index, row in df.iterrows():
		cascade_num, source, timestamp = row["cascade_number"], row["source"], row["timestamp"]
		if timestamp not in new_cascades[cascade_num]:
			new_cascades[cascade_num][timestamp] = list ()
		new_cascades[cascade_num][timestamp].append (source)

	return new_cascades	

def main (args):
	logging.basicConfig(
		filename=args.log_file,
		level=logging.INFO, 
		format= '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
		datefmt='%H:%M:%S'
	)
	# Started program log message
	logging.info (f'Script execution started')

	# Read the innovations
	with open (args.data_file, 'rb') as fin:
		idx, iidx, cascades, innovs = pickle.load (fin)

	logging.info (f"Randomizing cascades with {args.seed} seed")
	# Randomize the cascades
	randomized_cascades = randomize_cascades (cascades[0:args.ncascades], args.seed)
	logging.info (f"Randomized {len(randomized_cascades)} cascades")
	
	dims = len (idx)
	logging.info (f'Number of channels in the cascades: {dims}')

	result = hpmodels.DCHP.estimate (randomized_cascades, hpmodels.DCHP.log_likelihood_many_cascades, hpmodels.DCHP.log_likelihood_single_cascade, bandwidth=args.bandwidth, dims=dims, l2_coeff=args.l2_coeff, seed=42)
	
	with open(args.params_file,'wb') as f: pickle.dump(result, f)
	logging.info (f'Inference done, parameters written in file {args.params_file}')
	
if __name__ == "__main__":
	main (readArgs())
