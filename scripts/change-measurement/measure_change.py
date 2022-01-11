""" Measure change in word meaning and the time at which the change is most likely.
"""
import argparse
import numpy as np
import os
import logging
from tqdm import tqdm

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

def readArgs ():
	parser = argparse.ArgumentParser (description="Measure change per word")
	parser.add_argument ("--words-file", type=str, required=True, help="File contains the set of words to analyze")
	parser.add_argument ("--word-embeddings-dir", type=str, required=True, help="Directory contains all the aggregated embeddings")
	parser.add_argument ("--from-year", type=int, required=False, default=1990, help="Split from year")
	parser.add_argument ("--till-year", type=int, required=False, default=2019, help="Split till year")
	args = parser.parse_args ()
	return args

def standardize (X):
	return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

def compute_score (before_embedding, after_embedding, var, before_count, after_count):
	numerator = before_embedding - after_embedding
	denominator = var

	return np.dot ((np.sqrt(before_count) * numerator)/denominator, np.sqrt (after_count) * numerator)
	#return np.dot ((np.sqrt (before_count) * (before_embedding - after_embedding))/var, np.sqrt (after_count) * (before_embedding - after_embedding))

def compute_score2 (before_embedding, after_embedding, var):
	numerator = before_embedding - after_embedding
	denominator = var

	return np.dot (numerator/denominator, numerator)

def read_embeddings_from_file (filename):
	embeddings = list ()
	with open (filename) as fin:
		for line in fin:
			parts = line.strip().split ("\t")
			embedding = np.array(parts[-1].split()).astype(float)
			embeddings.append (embedding)
	return np.stack (embeddings, axis=0)

def main (args):
	words = set ()
	with open (args.words_file) as fin:
		for line in fin:
			words.add (line.strip())

	for word in tqdm(words):
		with open (os.path.join (args.word_embeddings_dir, word, f"{word}.computed_scores"), "w") as fout:
			# Get all the years for which we have embeddings.
			years = [year for year in range (args.from_year, args.till_year+1) if os.path.isfile (os.path.join (args.word_embeddings_dir, word, f"{year}.tsv"))]
			# Read the embeddings.
			embeddings_list = [read_embeddings_from_file (os.path.join (args.word_embeddings_dir, word, f"{year}.tsv")) for year in years]

			# Pool the embeddings
			before_embeddings = [np.concatenate (embeddings_list[0:i], axis=0).mean(axis=0) for i in range (1, len (embeddings_list))]
			after_embeddings = [np.concatenate (embeddings_list[i:], axis=0).mean(axis=0) for i in range (1, len (embeddings_list))]

			# Variance of the embeddings
			var = (np.concatenate (embeddings_list, axis=0)).var (axis=0)

			# Calculate the before and after counts
			before_counts = [np.concatenate (embeddings_list[0:i], axis=0).shape[0] for i in range (1, len (embeddings_list))]
			after_counts = [np.concatenate (embeddings_list[i:], axis=0).shape[0] for i in range (1, len (embeddings_list))]

			frequency_accounted_scores = [compute_score (before_embeddings[i], after_embeddings[i], var, before_counts[i], after_counts[i]) for i in range (len (before_embeddings))]
			scores = [compute_score2 (before_embeddings[i], after_embeddings[i], var) for i in range (len (before_embeddings))]
			i= 0
			for score, year in zip(scores, years[1:]):
				fout.write (f"{word}\t{year}\t{frequency_accounted_scores[i]}\t{score}\t{before_counts[i]}\t{after_counts[i]}\n")
				i+=1

if __name__ == "__main__":
	main (readArgs ())
