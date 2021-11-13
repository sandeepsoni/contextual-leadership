import argparse
import random
import pickle
import numpy as np
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
	parser.add_argument ("--bandwidth", type=float, required=False, default=1.0, help="Bandwidth parameter")
	parser.add_argument ("--seed", type=int, required=False, default=42, help="random seed")
	args = parser.parse_args ()
	return args

def main (args):
	# Started program log message
	logging.info (f'Script execution for optimization started')

	# Read the innovations
	with open (args.data_file, 'rb') as fin:
		idx, iidx, cascades, innovs = pickle.load (fin)
	
	dims = len (idx)
	logging.info (f'Number of channels in the cascades: {dims}')
	mu,b,c,s = hpmodels.DCHP.estimate (cascades, hpmodels.DCHP.log_likelihood_many_cascades, bandwidth=args.bandwidth, dims=dims, seed=args.seed)

	params = np.concatenate ((mu, b, c, s))
	log_likelihood = hpmodels.DCHP.log_likelihood_many_cascades (params, cascades, bandwidth=args.bandwidth, dims=dims, sign=1.0)
	logging.info (f'log likelihood: {log_likelihood}')
	
	with open(args.params_file,'wb') as f: pickle.dump((mu, b, c, s), f)
	logging.info (f'Coarsened HP inference done, parameters written in file {args.params_file}')
	
if __name__ == "__main__":
	main (readArgs())
