"""

The final list of change words is found by filtering out words with the following properties.

- The before and after counts should be at least 5 tokens per million tokens.

"""

import argparse
import logging
import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict

logging.basicConfig(
	format='%(asctime)s %(levelname)-8s %(message)s',
	level=logging.INFO,
	datefmt='%Y-%m-%d %H:%M:%S')

MILLION=1000000

def readArgs ():
	parser = argparse.ArgumentParser (description="Filter down the vocabulary")
	parser.add_argument ("--computed-scores-file", required=True, type=str, help="File contains a sorted list of words and their computed scores")
	parser.add_argument ("--word-embeddings-dir", required=True, type=str, help="File contains word embeddings and other statistics")
	parser.add_argument ("--min-frequency", required=False, type=int, default=5, help="Minimum frequency")
	parser.add_argument ("--from-year", required=False, type=int, default=1990, help="Minimum frequency")
	parser.add_argument ("--till-year", required=False, type=int, default=2019, help="Minimum frequency")	
	parser.add_argument ("--keep-file", required=True, type=str, help="Words that pass the filter")
	parser.add_argument ("--discard-file", required=True, type=str, help="Words that fail the filter")
	args = parser.parse_args ()
	return args

def main (args):
	# Read words from a vocab
	vocab = dict ()
	with open (args.computed_scores_file) as fin:
		for i, line in enumerate (tqdm(fin)):
			parts = line.strip().split ("\t")
			word = parts[0]
			year = int (parts[1])
			if word not in vocab:
				vocab[word] = year

	# Now for every word, get the before and after counts
	before_counts, after_counts = dict (), dict ()
	for word in vocab:
		filename = os.path.join (args.word_embeddings_dir, word, f"{word}.computed_scores")
		with open (filename) as fin:
			for line in fin:
				parts = line.strip().split ("\t")
				y = int (parts[1])
				if vocab[word] == y:
					before_count, after_count = int (parts[4]), int (parts[5])
					before_counts[word] = before_count
					after_counts[word] = after_count

	# Get overall count for every year
	overall_counts = dict ()
	for word in vocab:
		filename = os.path.join (args.word_embeddings_dir, word, f"{word}.overall_counts")
		with open (filename) as fin:
			for line in fin:
				parts = line.strip().split ("\t")
				y, count = int (parts[0]), int (parts[1])
				if y not in overall_counts:
					overall_counts[y] = 0
				overall_counts[y] += count	

	# Now apply the filter
	with open (args.keep_file, "w") as keep_file, open (args.discard_file, "w") as discard_file:
		for word in vocab:
			year = vocab[word]
			before_count = np.sum([overall_counts[y] for y in range (args.from_year, year)])
			after_count = np.sum ([overall_counts[y] for y in range (year, args.till_year + 1)])
			prop_before = (before_counts[word]/before_count) * MILLION
			prop_after = (after_counts[word]/after_count) * MILLION
			if prop_before >= args.min_frequency and prop_after >= args.min_frequency:
				keep_file.write (f"{word}\n")
			else:
				discard_file.write (f"{word}\n")

if __name__ == "__main__":
	main (readArgs ())
