"""

The final list of change words is found by filtering out words with the following properties.

- any word whose length is less than or equal to 2

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
	parser.add_argument ("--computed-scores-file", required=True, type=str, help="File contains a sorted list of words and their computed scores")
	parser.add_argument ("--min-length", required=False, type=int, default=2, help="Minimum frequency")
	parser.add_argument ("--keep-file", required=True, type=str, help="Words that pass the filter")
	parser.add_argument ("--discard-file", required=True, type=str, help="Words that fail the filter")
	args = parser.parse_args ()
	return args

def main (args):
	vocab = set ()
	with open (args.computed_scores_file) as fin:
		for i, line in enumerate (tqdm(fin)):
			parts = line.strip().split ("\t")
			word = parts[0]
			if word not in vocab:
				vocab.add (word)

	with open (args.keep_file, "w") as keep_file, open (args.discard_file, "w") as discard_file:
		for word in vocab:
			if len (word) > args.min_length:
				keep_file.write (f"{word}\n")
			else:
				discard_file.write (f"{word}\n")

if __name__ == "__main__":
	main (readArgs ())
