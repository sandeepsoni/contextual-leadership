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

def read_embedding_from_file (filename):
	with open (filename) as fin:
		embedding = np.array (fin.read().strip().split()).astype (float)

	return embedding

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
	if before_count > 0 and after_count > 0:
		before_embedding = before_embedding/before_count
		after_embedding = after_embedding/after_count
	elif before_count > 0:
		before_embedding = before_embedding/before_count
	elif after_count > 0:
		after_embedding = after_embedding/after_count

	return before_embedding, after_embedding
	

def rescale_embeddings (embeddings, counts):
	return {y: counts[y] * embeddings[y] for y in embeddings}

def compute_score (before_count, before_embedding, after_count, after_embedding, var_embedding):
	numerator = (np.sqrt (before_count) * before_embedding) - (np.sqrt (after_count) * after_embedding)
	denominator = var_embedding
	return (numerator/denominator).dot (numerator)

def main (args):
	words = set ()
	with open (args.words_file) as fin:
		for line in fin:
			words.add (line.strip())

	for word in words:
		with open (os.path.join (args.word_embeddings_dir, word, f"{word}.computed_scores"), "w") as fout:
			# Read the counts file as dictionary.
			counts = read_counts_from_file (os.path.join (args.word_embeddings_dir, word, f"{word}.overall_counts"))
			embeddings = read_embeddings_from_files (os.path.join (args.word_embeddings_dir, word), args.from_year, args.till_year)
			sum_embeddings = rescale_embeddings (embeddings, counts)	
			scores = dict ()
			cumulative_counts = dict ()
			for split_year in range (args.from_year, args.till_year):
				before_count, after_count = split_counts (split_year, counts)
				cumulative_counts[split_year] = (before_count, after_count)
				before_embedding, after_embedding = split_embeddings (sum_embeddings, split_year, before_count, after_count)
				var_embedding = read_embedding_from_file (os.path.join (args.word_embeddings_dir, word, f"{word}.overall_var_embedding"))
				score = compute_score (before_count, before_embedding, after_count, after_embedding, var_embedding)
				scores[split_year] = score

			max_score = max ([(y,score) for y,score in scores.items()], key=lambda x:x[1])
			fout.write (f"{word}\t{max_score[0]}\t{cumulative_counts[max_score[0]][0]}\t{cumulative_counts[max_score[0]][1]}\t{max_score[1]}\n")

if __name__ == "__main__":
	main (readArgs ())
