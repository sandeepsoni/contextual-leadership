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

def read_counts_from_file (filename):
	counts = dict ()
	with open (filename) as fin:
		for line in fin:
			parts = line.strip().split ("\t")
			year, count = int (parts[0]), int (parts[1])
			counts[year] = count

	return counts

def read_embeddings_from_files (dirname, from_year, till_year):
	embeddings = dict ()
	for year in range (from_year, till_year+1):
		filename = os.path.join (dirname, f"{year}.mean_embedding")
		with open (filename) as fin:
			embeddings[year] = np.array (fin.read().strip().split()).astype (float)

	return embeddings

def split_counts (year, counts):
	years = sorted (list (counts.keys()))
	before_count = sum([counts[y] for y in years if y <= year])
	after_count = sum([counts[y] for y in years if y > year])
	return before_count, after_count	

def split_embeddings (sum_embeddings, year, before_count, after_count):
	years = sorted (list (sum_embeddings.keys()))
	before_embedding = np.stack ([sum_embeddings[year] for y in years if y <= year])
	after_embedding = np.stack ([sum_embeddings[year] for y in years if y > year])
	before_embedding = before_embedding.sum (axis=0)
	after_embedding = after_embedding.sum (axis=0)
	return before_embedding/before_count, after_embedding/after_count

def rescale_embeddings (embeddings, counts):
	return {y: counts[y] * embeddings[y] for y in embeddings}

def main (args):
	words = set ()
	with open (args.words_file) as fin:
		for line in fin:
			words.add (line.strip())

	for word in words:
		# Read the counts file as dictionary.
		counts = read_counts_from_file (os.path.join (args.word_embeddings_dir, word, f"{word}.overall_counts"))
		embeddings = read_embeddings_from_files (os.path.join (args.word_embeddings_dir, word), args.from_year, args.till_year)
		sum_embeddings = rescale_embeddings (embeddings, counts)	
		for split_year in range (args.from_year, args.till_year):
			before_count, after_count = split_counts (split_year, counts)
			before_embedding, after_embedding = split_embeddings (sum_embeddings, split_year, before_count, after_count)
			print (word, year, before_count, after_count, " ".join(list (map(str, before_embedding.tolist())))," ".join(list (map(str, after_embedding.tolist()))))

if __name__ == "__main__":
	main (readArgs ())
