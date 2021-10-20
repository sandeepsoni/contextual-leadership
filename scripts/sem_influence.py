import pickle
import sys
sys.path.append ("../modules")
import hpmodels 

import logging
logging.basicConfig(
	level=logging.INFO, 
	format= '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
	datefmt='%H:%M:%S'
)

def main ():
	# Constants
	data_file = "../data/data-v1.001.pkl"
	params_file = "../data/params-v1.001.pkl"

	# Started program log message
	logging.info (f'Script execution for optimization started')

	# Read the innovations
	with open (data_file, 'rb') as fin:
		idx, iidx, cascades, innovs = pickle.load (fin)
	
	dims = len (idx)
	logging.info (f'Number of channels in the cascades: {dims}')
	mu,b,c,s = hpmodels.DCHP.estimate (cascades, hpmodels.DCHP.log_likelihood_many_cascades, dims=dims)
	
	with open(params_file,'wb') as f: pickle.dump((mu, b, c, s), f)
	logging.info (f'Coarsened HP inference done, parameters written in file {params_file}')
	
if __name__ == "__main__":
	main ()
