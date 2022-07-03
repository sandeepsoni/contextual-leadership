import sklearn
from sklearn import linear_model
import argparse
import pandas as pd
from scipy import sparse
import numpy as np
from tqdm import tqdm
import json
import os

def readArgs ():
	parser = argparse.ArgumentParser (description="Run a regression model on the counts data for papers")
	parser.add_argument ("--paper-ids-file", type=str, required=True, help="File contains paper ids and other metadata")
	parser.add_argument ("--input-file", type=str, required=True, help="File contains counts of innovations for each paper")
	parser.add_argument ("--history-window", type=int, required=False, default=30, help="the history window that should be considered")
	parser.add_argument ("--publication-year", type=int, required=False, default=2014, help="Consider this year to be the publication year whose regression coefficients we are interested in")
	parser.add_argument ("--bandwidth", type=float, required=False, default=1.0, help="time scale parameter of the kernel")
	parser.add_argument ("--regularization", type=float, required=False, default=0.0, help="Regularization penalty")
	parser.add_argument ("--coefficients-file", type=str, required=True, help="File contains base rate and linguistic coefficients")
	parser.add_argument ("--inference-period", type=int, required=False, default=2, help="The inference period; our regression runs till publication year+inference period")
	args = parser.parse_args ()
	return args

def read_paper_ids_from_file (filename, year=2016):
	""" Read the paper ids from a JSONL file. 
	    Our regression happens at the end of the inference period so we should 
	    consider only those papers that have been published before the end of the inference period.	    

	Parameters:
	===========
	filename (str): The path of the JSONL file.
	year (int): The year up to which papers are considered (default: 2016)
	
	Returns:
	========
	idx, iidx (tuple): idx is a dictionary that maps paper_ids to numeric ids,
			   iidx is a dictionary that maps numeric ids to paper_ids
	"""

	paper_ids = set ()
	with open (filename) as fin:
		for line in fin:
			js = json.loads (line.strip())
			y = js["metadata"]["year"]
			if y <= year:
				paper_ids.add (int (js["paper_id"]))

	idx = {paper_id: i for i, paper_id in enumerate (paper_ids)}
	iidx = {i: paper_id for i, paper_id in enumerate (paper_ids)}
	
	return (idx, iidx)
 
def read_innovations_from_file (filename, year=2014):
	""" Read the innovations from file and create a mapping of all the innovations.
	
	Our regression runs at the end of the inference period, so we should only consider
	innovations that appear at the beginning of the inference period.

	Parameters:
	===========
	filename (str): The path of the filename that contains the innovations.

	Returns:
	========
	idx, iidx (tuple): idx contains a map of innovation to numeric id,
			   iidx contains a map of numeric id to innovation.
	"""
	innovs = set ()
	with open (filename) as fin:
		for line in fin:
			js = json.loads (line.strip())
			y = js["year"]
			if y <= year:
				innovs.add (js["word"])

	idx = {word:i for i, word in enumerate (innovs)}
	iidx = {i:word for i, word in enumerate (innovs)}
	
	return (idx, iidx)

def read_file_as_sparse_matrix (filename, papers_index, innovs_index, bandwidth=1.0, year=2016, history_window=30):
	""" Read from the counts file and convert into features. Also get the ground truth.
	    We expect the year to be the end of the inference period.
	    
	Parameters:
	===========

	filename (str): The path of the filename from which to read the counts and convert 
			them into features.
	Returns:
	========
	X (scipy.sparse): Sparse matrix representation of features.
	y (np.ndarray): Ground truth of counts.
	"""

	kernel_expansions = {i: np.exp (-i * bandwidth) for i in range (100)}
	pidx, piidx = papers_index
	idx, iidx = innovs_index
	nrows = len (pidx) * len (idx)
	ncols = len (pidx)
	X = sparse.lil_matrix ((nrows, 2*ncols))
	y = np.zeros (nrows)

	with open (filename) as fin:
		for line in tqdm (fin):
			js = json.loads (line.strip())
			word = js["word"]
			paper_id = js["paper_id"]
			yr = js["year"]
			num_innovations = js["num_innovations"]
			if yr <= year and word in idx and paper_id in pidx:
				row = pidx[paper_id] * idx[word]
				col = pidx[paper_id]
				X[row, col] = 1.0
				y[row] = num_innovations
				offset = len (pidx)
				for item in js["previous_papers"]:
					pid = item["paper_id"]
					t = item["year"]
					if history_window >= (yr-t):
						X[row, offset + pidx[pid]] = kernel_expansions[int (yr - t)]

	return X.tocsr (), y

def read_years_from_file (filename):
	""" Read the publication year of each paper from file and map into a dictionary.

	Parameters:
	===========
	filename (str): Path to the file that contains the publication year.

	Returns:
	========
	paper2year (dict): Mapping of paper_ids to their publication year.

	"""
	paper2year = dict ()
	with open (filename) as fin:
		for line in fin:
			js = json.loads (line.strip())
			y = js["metadata"]["year"]
			paper_id = int(js["paper_id"])
			paper2year[paper_id] = y

	return paper2year

def main (args):
	dirname = os.path.dirname (args.coefficients_file)
	os.makedirs (dirname, exist_ok=True)
	
	# 1. Collect all the papers that appear till the end of the inference period
	papers_index = read_paper_ids_from_file (args.paper_ids_file, year=args.publication_year + args.inference_period)

	# 2. Map the publication years of all the papers (this can be used to look up later)
	paper2years = read_years_from_file (args.paper_ids_file)

	# 3. Collect all the innovations up to the start of the inference period i.e at the publication year.
	innovs_index = read_innovations_from_file (args.input_file, year=args.publication_year)

	# 4. Collect features till the end of the inference period.
	X,y = read_file_as_sparse_matrix (args.input_file, papers_index, innovs_index, bandwidth=args.bandwidth, year=args.publication_year + args.inference_period, history_window=args.history_window)

	# 5. Run the regression model and collect all the coefficients (We'll later filter out only those coefficients that are from the same publication period)
	clf = linear_model.PoissonRegressor(fit_intercept=False, alpha=args.regularization, max_iter=500, tol=1e-6, verbose=3)
	clf.fit(X, y)
	print (f"Deviance: {clf.score(X,y)}")
	coeffs = clf.coef_
	coefficients = list ()
	idx, iidx = papers_index
	for i in sorted (iidx.keys()):
		paper_id = iidx[i]
		base_rate = coeffs[i]
		influence = coeffs[len(iidx) + i]
		y = paper2years[paper_id]
		coefficients.append ([paper_id, y, base_rate, influence, y==args.publication_year])

	output = pd.DataFrame (coefficients, columns=["paper_id", "publication_year", "base_rate", "influence", "valid_coeff"])
	output.to_csv (args.coefficients_file, sep="\t", index=False, header=True)

if __name__ == "__main__":
	main (readArgs ())
