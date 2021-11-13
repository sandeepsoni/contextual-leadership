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
	parser.add_argument ("--params-file", type=str, required=True, help="The name of the file containing all the params")
	parser.add_argument ("--bandwidth", type=float, required=False, default=2.0, help="Bandwidth parameter")
	parser.add_argument ("--seed", type=int, required=False, default=42, help="random seed")
	#parser.add_argument ("--train-percent", type=float, required=False, default=0.8, help="train percentage in a train-test split")
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
	# Started program log message
	logging.info (f'Script execution started')

	# Read the innovations
	with open (args.data_file, 'rb') as fin:
		idx, iidx, cascades, innovs = pickle.load (fin)

	# Randomize the cascades
	randomized_cascades = randomize_cascades (cascades, args.seed)

	# Divide the cascades into a random train and dev
	#random.seed (args.seed)
	#random.shuffle (cascades)
	#random.seed (args.seed)
	#random.shuffle (innovs)
	#train_cascades = cascades[:int((len(cascades)+1)*args.train_percent)]
	#test_cascades = cascades[int((len(cascades)+1)*args.train_percent):] 
	
	dims = len (idx)
	logging.info (f'Number of channels in the cascades: {dims}')
	#mu,b,c,s = hpmodels.DCHP.estimate (train_cascades, hpmodels.DCHP.log_likelihood_many_cascades, bandwidth=args.bandwidth, dims=dims)

	#params = np.concatenate ((mu, b, c, s))
	#heldout_log_likelihood = hpmodels.DCHP.log_likelihood_many_cascades (params, test_cascades, bandwidth=args.bandwidth, dims=dims, sign=1.0)	
	#logging.info (f'Held out log likelihood: {heldout_log_likelihood}')

	mu,b,c,s = hpmodels.DCHP.estimate (randomized_cascades, hpmodels.DCHP.log_likelihood_many_cascades, bandwidth=args.bandwidth, dims=dims)
	
	with open(args.params_file,'wb') as f: pickle.dump((mu, b, c, s), f)
	logging.info (f'Coarsened HP inference done, parameters written in file {args.params_file}')
	
if __name__ == "__main__":
	main (readArgs())
