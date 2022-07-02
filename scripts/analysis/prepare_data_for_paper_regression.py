"""
python prepare_data_for_paper_regression.py --citation-file /global/scratch/users/sandeepsoni/projects/hp-modeling/data/raw/s2orc_acl.citation_years.jsonl --sem-coefficients-dir /global/scratch/users/sandeepsoni/projects/hp-modeling/data/experiments/004/paper_coefficients/ --lex-coefficients-dir /global/scratch/users/sandeepsoni/projects/hp-modeling/data/experiments/004/paper_coefficients_for_lex_innovs/ --venues-file /global/scratch/users/sandeepsoni/projects/hp-modeling/data/raw/venues_shortform.tsv --output-file /global/scratch/users/sandeepsoni/projects/hp-modeling/data/analysis/papers_regression_raw_data.tsv
"""


import argparse
import json
import glob
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from functools import reduce

def readArgs ():
	parser = argparse.ArgumentParser (description="Prepare data for regression")
	parser.add_argument ("--citation-file", type=str, required=True, help="File contains paper ids, publication year and the citation distribution")
	parser.add_argument ("--sem-coefficients-dir", type=str, required=True, help="Directory that contains linguistic coefficients for semantic influence")
	parser.add_argument ("--lex-coefficients-dir", type=str, required=True, help="Directory that contains linguistic coefficients for linguistic influence")
	parser.add_argument ("--topics-file", type=str, required=True, help="File contains the topics for each paper")
	parser.add_argument ("--venues-file", type=str, required=True, help="File contains the venues for each paper")
	parser.add_argument ("--output-file", type=str, required=True, help="File contains the prepared data for subsequent regression")
	parser.add_argument ("--num-topics", type=int, required=False, default=10, help="The number of topics")
	parser.add_argument ("--k", type=int, required=False, default=2, help="offset in years")
	parser.add_argument ("--n", type=int, required=False, default=5, help="test of time offset in years")
	args = parser.parse_args ()
	return args

def read_topics_as_df (filename, num_topics=10, index_col="paper_id", prefix="topic_"):
	with open (filename) as fin:
		lines = [line for line in tqdm(fin)]
	mat = np.zeros ((len (lines), num_topics))
  	
	paper2rownum = dict ()
	for line in lines:
		js = json.loads (line) 
		paper_id = js["paper_id"]
		if paper_id not in paper2rownum:
			paper2rownum[paper_id] = len (paper2rownum)
		for topic in js["topics"]:
			topic_id = topic[0]
			prob = topic[1]
			mat[paper2rownum[paper_id], topic_id] = float (prob)

	rows = list ()
	for paper in paper2rownum:
		rows.append ([paper] + mat[paper2rownum[paper],:].tolist())
	header = [index_col] + [prefix+str(i) for i in range (mat.shape[1])]
	df = pd.DataFrame (rows, columns=header)
	return df

def read_ling_coefficients_as_df (dirname, k=2, prefix="001", output_col_name="sem_influence"):
	""" Read the linguistic influence estimates from file as a dataframe.
	
	Parameters:
	===========
	dirname (str): The path to the directory that contains the files with coefficient estimates.
	k (int): The number of years post publication that is the inference period (defaults: 2)
	prefix (str): Prefix of the files to consider.
	output_col_name (str): The column name in the dataframe under which you'll find the estimates.
	
	Returns:
	========
	df (pandas.DataFrame): The dataframe that contains the coefficient estimates of linguistic influence.

	"""

	coefficients = list ()
	files = glob.glob (os.path.join (dirname, "*.tsv"))
	for filename in tqdm (files):
		basename = os.path.basename (filename)
		if basename.startswith (prefix):
			ling_df = pd.read_csv (filename, sep="\t")
			title = basename.split (".")[0]
			year = int (title[-4:])
			search_year = year #- k
			ling_df = ling_df[ling_df.publication_year == search_year]
			for _, row in ling_df.iterrows():
				coefficients.append ([str(int (row["paper_id"])), row["influence"]])

	return pd.DataFrame (coefficients, columns=["paper_id", output_col_name])

def read_citations_as_df (filename, k=2, n=5):
	""" Read the citation statistics from filename.

	Parameters:
	===========
	filename (str): The path of the file that contains the citations.
	k(int): The number of years in the inference period post publication (default:2)
	n(int): The number of years in the prediction period post publication (default:5)

	Returns:
	========
	df (pandas.DataFrame): The dataframe contains relevant columns and all the statistics.
	"""

	rows = list ()
	with open (filename) as fin:
		for line in fin:
			js = json.loads (line)
			paper_id = js["paper_id"]
			year = int (js["publication_year"])
			k_cites = sum ([js["citation_distribution"].get (str(y), 0) for y in range (year+1, year+k+1)])
			n_cites = sum ([js["citation_distribution"].get (str(y), 0) for y in range (year+1, year+n+1)])
			rows.append ([paper_id, year, np.log (k_cites+1), np.log (n_cites+1)])

	df = pd.DataFrame (rows, columns=["paper_id", "publication_year", f"logk_cites(k={k})", f"logn_cites(n={n})"])
	return df

def main (args):
	cites_df = read_citations_as_df (args.citation_file, k=args.k, n=args.n)
	sem_coeffs_df = read_ling_coefficients_as_df (args.sem_coefficients_dir, k=args.k, output_col_name="sem_influence") 
	lex_coeffs_df = read_ling_coefficients_as_df (args.lex_coefficients_dir, k=args.k, output_col_name="lex_influence")
	topics_df = read_topics_as_df (args.topics_file, num_topics=args.num_topics)
	venues_df = pd.read_csv (args.venues_filename, sep="\t")
	dataframes = [cites_df, sem_coeffs_df, lex_coeffs_df, topics_df, venues_df]
	overall_df = reduce (lambda left,right: pd.merge (left, right, how="inner", on="paper_id"), dataframes)
	overall_df.to_csv (args.output_file, sep="\t", header=True, index=False)

if __name__ == "__main__":
	main (readArgs ())
