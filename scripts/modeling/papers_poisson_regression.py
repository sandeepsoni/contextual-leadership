import sklearn
from sklearn import linear_model
import argparse
import pandas as pd
from scipy import sparse
import numpy as np
from tqdm import tqdm
import json

def readArgs ():
	parser = argparse.ArgumentParser (description="Run a regression model on the counts data for papers")
	parser.add_argument ("--paper-ids-file", type=str, required=True, help="File contains paper ids and other metadata")
	parser.add_argument ("--input-file", type=str, required=True, help="File contains counts of innovations for each paper")
	parser.add_argument ("--regularization", type=float, required=False, default=0.0, help="Regularization penalty")
	parser.add_argument ("--coefficients-file", type=str, required=True, help="File contains base rate and linguistic coefficients")
	args = parser.parse_args ()
	return args

def read_paper_ids_from_file (filename):
	paper_ids = set ()
	with open (filename) as fin:
		for line in fin:
			js = json.loads (line.strip())
			paper_ids.add (int (js["paper_id"]))

	idx = {paper_id: i for i, paper_id in enumerate (paper_ids)}
	iidx = {i: paper_id for i, paper_id in enumerate (paper_ids)}
	
	return (idx, iidx)
 
def read_innovations_from_file (filename):
	innovs = set ()
	with open (filename) as fin:
		for line in fin:
			js = json.loads (line.strip())
			innovs.add (js["word"])

	idx = {word:i for i, word in enumerate (innovs)}
	iidx = {i:word for i, word in enumerate (innovs)}
	
	return (idx, iidx)

def read_file_as_sparse_matrix (filename, papers_index, innovs_index):
	kernel_expansions = {i: np.exp (-i) for i in range (100)}
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
			year = js["year"]
			num_innovations = js["num_innovations"]
			row = pidx[paper_id] * idx[word]
			col = pidx[paper_id]
			X[row, col] = 1.0
			y[row] = num_innovations
			offset = len (pidx)
			for item in js["previous_papers"]:
				pid = item["paper_id"]
				t = item["year"]
				X[row, offset + pidx[pid]] = kernel_expansions[int (year - t)]

	return X.tocsr (), y	

def main (args):
	papers_index = read_paper_ids_from_file (args.paper_ids_file)
	innovs_index = read_innovations_from_file (args.input_file)
	X,y = read_file_as_sparse_matrix (args.input_file, papers_index, innovs_index)

	clf = linear_model.PoissonRegressor(fit_intercept=False, alpha=args.regularization, tol=1e-6, verbose=3)
	clf.fit(X, y)
	print (clf.score(X,y))
	coeffs = clf.coef_
	coefficients = list ()
	idx, iidx = papers_index
	for i in sorted (iidx.keys()):
		paper_id = iidx[i]
		base_rate = coeffs[i]
		influence = coeffs[len(iidx) + i]
		coefficients.append ([paper_id, base_rate, influence])

	output = pd.DataFrame (coefficients, columns=["paper_id", "base_rate", "influence"])
	output.to_csv (args.coefficients_file, sep="\t", index=False, header=True)	

if __name__ == "__main__":
	main (readArgs ())
	
