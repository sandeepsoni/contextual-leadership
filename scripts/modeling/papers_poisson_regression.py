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
	parser.add_argument ("--input-file", type=str, required=True, help="File contains counts of innovations for each paper")
	parser.add_argument ("--coefficients-file", type=str, required=True, help="File contains base rate and linguistic coefficients")
	args = parser.parse_args ()
	return args

def read_paper_ids_from_file (filename):
	paper_ids = set ()
	with open (filename) as fin:
		for line in fin:
			js = json.loads (line.strip())
			paper_ids.add (js["paper_id"])

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

def df_to_sparse (df, kernel_expansions):
	def create_index (items):
		items = set (items)
		idx = {item: i for i, item in enumerate (items)}
		iidx = {i:item for i, item in enumerate (items)}
		return (idx, iidx)

	innovs_idx, innovs_iidx = create_index ([item for item in df.word])
	papers_idx, papers_iidx = create_index ([item for item in df.paper_id])

	# create empty sparse matrix
	X = sparse.dok_matrix((len(papers_idx)*len(innovs_idx), 2*len(papers_idx)), dtype=np.float32)
	y = np.zeros (len(papers_idx)*len(innovs_idx))
	for word in tqdm(innovs_idx):
		individual_word_df = df[df.word == word]
		for _, row in individual_word_df.iterrows():
			innov = row["word"]
			year = row["year"]
			paper_id = row["paper_id"]
			count = row["num_innovations"]
			history = individual_word_df[individual_word_df.year < year]

			row_num = innovs_idx[innov] * len (innovs_idx) + papers_idx[paper_id]
			X[row_num,papers_idx[paper_id]] = 1 # base rate term
			for _, past_row in history.iterrows():
				past_paper_id = past_row["paper_id"]
				past_year = past_row["year"]
				X[row_num, len(papers_idx) + papers_idx[past_paper_id]] = kernel_expansions[year - past_year] # influence features

			y[row_num] = count

	return (innovs_idx, innovs_iidx), (papers_idx, papers_iidx), X.to_csr (),y

def main (args):
	papers_index = read_paper_ids_from_file (args.input_file)
	innovs_index = read_innovations_from_file (args.input_file)
	return

	paper_counts = pd.read_csv (args.counts_file, sep="\t", names=["word", "year", "paper_id", "num_innovations"])
	kernel_expansions = {i: np.exp (-i) for i in range (100)}
	innovs_index, papers_index, X,y = df_to_sparse (paper_counts, kernel_expansions)
	clf = linear_model.PoissonRegressor(fit_intercept=False, alpha=0, tol=1e-6, verbose=3)
	clf.fit(X, y)
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
	
