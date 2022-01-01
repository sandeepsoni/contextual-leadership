"""

Before computing the first and the second moment, we filter down the vocabulary by keeping words that:

- are alphabetic

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
	parser.add_argument ("--min-freq", required=False, type=int, default=30, help="Minimum frequency")
	parser.add_argument ("--keep-file", required=True, type=str, help="Words that pass the filter")
	parser.add_argument ("--discard-file", required=True, type=str, help="Words that fail the filter")
	args = parser.parse_args ()
	return args

def main (args):
	vocab = set ()
	with open (args.input_embeddings_file) as fin:
		for i, line in enumerate (tqdm(fin)):
			parts = line.strip().split ("\t")
			word = parts[3]
			if word not in vocab:
				vocab.add (word)

	with open (args.keep_file, "w") as keep_file, open (args.discard_file, "w") as discard_file:
		for word in vocab:
			if word.isalpha():
				keep_file.write (f"{word}\n")
			else:
				discard_file.write (f"{word}\n")

if __name__ == "__main__":
	main (readArgs ())
