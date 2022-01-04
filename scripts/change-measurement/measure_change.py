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

def readEmbeddingsFromFile (filename, sep='\t'):
	years = list ()
	embeddings = list ()
	with open (filename) as fin:
		for line in fin:
			parts = line.split (sep)
			year, embedding = parts[1], parts[-1].split ()
			year = int (year)
			embedding = np.array (embedding).astype (float)
			years.append (year)
			embeddings.append (embedding)

	years = np.array (years)
	embeddings = np.array (embeddings)

	indices = np.argsort (years)
	years = years[indices]
	embeddings = embeddings[indices, :]
	return years, embeddings

def standardize (X):
	return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

def boundary_points (array):
	points = dict ()
	for i, elem in enumerate (array):
		points[elem] = i+1
	return points

def compute_score (X, t):
	n = X.shape[0]
	numerator = (np.sqrt (t) * X[:t].mean(axis=0)) - (np.sqrt (n-t) * X[t:].mean(axis=0))
	denominator = X.var(axis=0)
	return (numerator / denominator).dot(numerator)

def main (args):
	years, embeddings = readEmbeddingsFromFile (args.embeddings_file)
	embeddings = standardize (embeddings)
	split_indices = boundary_points (years)
	split_points = sorted (list(split_indices.keys()))[:-1]
	scores = [compute_score (embeddings, split_indices[split]) for split in split_points]

	with open (args.output_file, "w") as fout:
		for i, score in enumerate (scores):
			fout.write(f'{split_points[i]},{score}\n')


def main (args):
	words = set ()
	with open (args.words_file) as fin:
		for line in fin:
			words.add (line.strip())

	for word in words:
		# Read the counts file as dictionary.
		counts = dict ()
		with open (os.path.join (args.word_embeddings_dir, f"{word}.overall_counts")) as fin:
			for line in fin:
				parts = line.strip().split ("\t")
				year, count = int (parts[0]), int (parts[1])
				counts[year] = count

			logging.info (f"{word}: {sum(list (counts.values()))}")
					

if __name__ == "__main__":
	main (readArgs ())
