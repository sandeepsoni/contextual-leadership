import argparse
import pandas as pd
import json
from tqdm import tqdm

def readArgs ():
	parser = argparse.ArgumentParser (description="Collect all the data for analysis")
	parser.add_argument ("--coefficients-file", type=str, required=True, help="coefficients file")
	parser.add_argument ("--citations-file", type=str, required=True, help="file contains papers metadata")
	parser.add_argument ("--output-file", type=str, required=True, help="file contains output")
	args = parser.parse_args ()
	return args

def main (args):
	coefficients = dict ()
	with open (args.coefficients_file) as fin:
		for i, line in enumerate (tqdm (fin)):
			if i > 0:
				parts = line.strip().split ("\t")
				paper_id = parts[0]
				coefficient = float (parts[2])
				coefficients[paper_id] = coefficient

	citations = dict ()
	with open (args.citations_file) as fin:
		for i, line in enumerate (tqdm (fin)):
			js = json.loads (line.strip())
			paper_id = js["paper_id"]
			cites = len (js["inlinks"])
			citations[paper_id] = cites

	

	rows = list ()
	for paper_id in citations:
		if paper_id in coefficients:
			rows.append ([paper_id, coefficients[paper_id], citations[paper_id]])

	df = pd.DataFrame (rows, columns=["paper_id", "ling_coeff", "n_cites"])
	df.to_csv (args.output_file, sep="\t", header=True, index=False)


if __name__ == "__main__":
	main (readArgs ())
