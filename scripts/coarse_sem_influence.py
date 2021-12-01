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
	parser.add_argument ("--train-percent", type=float, required=False, default=0.8, help="train percentage in a train-test split")
	parser.add_argument ("--l2-coeff", type=float, required=False, default=0.0, help="L2 coefficient for the regularization")
	args = parser.parse_args ()
	return args

def main (args):
	# Started program log message
	logging.info (f'Script execution for optimization started')

	# Read the innovations
	with open (args.data_file, 'rb') as fin:
		idx, iidx, cascades, innovs = pickle.load (fin)

	# Divide the cascades into a random train and dev
	random.seed (args.seed)
	random.shuffle (cascades)
	random.seed (args.seed)
	random.shuffle (innovs)
	train_cascades = cascades[:int((len(cascades)+1)*args.train_percent)]
	test_cascades = cascades[int((len(cascades)+1)*args.train_percent):] 
	
	dims = len (idx)
	logging.info (f'Number of channels in the cascades: {dims}')
	result = hpmodels.DCHP.estimate (train_cascades, hpmodels.DCHP.log_likelihood_many_cascades, hpmodels.DCHP.log_likelihood_single_cascade, bandwidth=args.bandwidth, dims=dims, l2_coeff=args.l2_coeff)

	heldout_log_likelihood = hpmodels.DCHP.log_likelihood_many_cascades (result.x, test_cascades, hpmodels.DCHP.log_likelihood_single_cascade, bandwidth=args.bandwidth, dims=dims, sign=1.0)
	logging.info (f'Held out log likelihood: {heldout_log_likelihood}')

	#result = hpmodels.DCHP.estimate (cascades, hpmodels.DCHP.log_likelihood_many_cascades, hpmodels.DCHP.log_likelihood_single_cascade, bandwidth=args.bandwidth, dims=dims)
	
	with open(args.params_file,'wb') as f: pickle.dump(result, f)
	logging.info (f'Coarsened HP inference done, parameters written in file {args.params_file}')
	
if __name__ == "__main__":
	main (readArgs())
