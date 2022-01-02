"""

Before computing the first and the second moment, we filter down the vocabulary by keeping words that:

- are present at least 25% of all the papers

"""

import argparse
import logging
from tqdm import tqdm
from collections import defaultdict

logging.basicConfig(
	format='%(asctime)s %(levelname)-8s %(message)s',
	level=logging.INFO,
	datefmt='%Y-%m-%d %H:%M:%S')

def readArgs ():
	parser = argparse.ArgumentParser (description="Filter down the vocabulary")
	parser.add_argument ("--input-embeddings-file", required=True, type=str, help="Single embeddings file")
	parser.add_argument ("--min-year-fraction", required=False, type=float, default=0.25, help="Minimum frequency")
	parser.add_argument ("--keep-file", required=True, type=str, help="Words that pass the filter")
	parser.add_argument ("--discard-file", required=True, type=str, help="Words that fail the filter")
	args = parser.parse_args ()
	return args

def main (args):
	vocab = defaultdict (set)
	years = set ()
	with open (args.input_embeddings_file) as fin:
		for i, line in enumerate (tqdm(fin)):
			parts = line.strip().split ("\t")
			word = parts[3]
			year = int(parts[1])
			years.add (year)
			vocab[word].add (year)

	total_years = len (years)
	with open (args.keep_file, "w") as keep_file, open (args.discard_file, "w") as discard_file:
		for word in vocab:
			if len(vocab[word])/total_years >= args.min_year_fraction:
				keep_file.write (f"{word}\n")
			else:
				discard_file.write (f"{word}\n")

if __name__ == "__main__":
	main (readArgs ())
