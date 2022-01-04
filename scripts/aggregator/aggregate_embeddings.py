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
	parser = argparse.ArgumentParser (description="Aggregate the counts and embeddings for each word")
	parser.add_argument ("--words-file", type=str, required=True, help="File contains the words to process")
	parser.add_argument ("--word-embeddings-dir", type=str, required=True, help="Directory contains embedding subdirectories for each word")
	parser.add_argument ("--from-year", type=int, required=False, default=1990, help="The first year in our corpus")
	parser.add_argument ("--till-year", type=int, required=False, default=2019, help="The last year in our corpus")
	parser.add_argument ("--dims", type=int, required=False, default=3072, help="Dimensionality of the embeddings")
	args = parser.parse_args ()
	return args

# copied from https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
# which works well even for tensor inputs
def update_moments (existing_statistics, new_value):
	""" Computes the running mean and variance. """
	(count, mean, M2)= existing_statistics
	count += 1
	delta = new_value - mean
	mean += delta / count
	delta2 = new_value - mean
	M2 += delta * delta2

	return (count, mean, M2)

def finalize_moments (existing_statistics):
	""" Calculate the first and second moments from the existing sum. """
	(count, mean, M2) = existing_statistics
	return (mean, M2 / count)


def main (args):
	words = set ()
	with open (args.words_file) as fin:
		for line in fin:
			words.add (line.strip())

	for i, word in enumerate (words):
		word_dir = os.path.join (args.word_embeddings_dir, word)
		overall_counts = dict ()
		overall_count = 0
		overall_mean_embedding = np.zeros (args.dims)
		overall_var_embedding = np.zeros (args.dims)
		for year in range (args.from_year, args.till_year+1):
			filename = os.path.join (word_dir, f"{year}.tsv")
			count = 0
			mean_embedding = np.zeros (args.dims)
			var_embedding = np.zeros (args.dims)
			if os.path.isfile (filename):
				with open (filename) as fin:
					for line in fin:
						parts = line.strip().split ("\t")
						embedding = np.array (list (map (float, parts[-1].split())))
						count, mean_embedding, var_embedding = update_moments ((count,mean_embedding,var_embedding), embedding)
						overall_count, overall_mean_embedding, overall_var_embedding = update_moments ((overall_count, overall_mean_embedding, overall_var_embedding), \
																									   embedding)
						
					mean_embedding, var_embedding = finalize_moments ((count, mean_embedding, var_embedding))

			with open (os.path.join (word_dir, f"{year}.counts"), "w") as fout:
				fout.write (f"{count}\n")

			with open (os.path.join (word_dir, f"{year}.mean_embedding"), "w") as fout:
				fout.write (f"{list (map(str, mean_embedding.tolist()))}\n")
				
			with open (os.path.join (word_dir, f"{year}.var_embedding"), "w") as fout:
				fout.write (f"{list (map(str, var_embedding.tolist()))}\n")

			logging.info (f"Statistics for {word} in {year} written")

			overall_counts[year] = count

		overall_mean_embedding, overall_var_embedding = finalize_moments ((overall_count, overall_mean_embedding, overall_var_embedding))

		if (i+1) % 500 == 0:
			logging.info (f"Processed {i+1} words from {args.words_file}")

		with open (os.path.join (word_dir, f"{word}.overall_counts"), "w") as fout:
			for year in sorted (overall_counts.keys()):
				fout.write (f"{year}\t{overall_counts[year]}\n")

		with open (os.path.join (word_dir, f"{word}.overall_mean_embedding"), "w") as fout:
			fout.write (f"{list (map(str, overall_mean_embedding.tolist()))}\n")
				
		with open (os.path.join (word_dir, f"{word}.overall_var_embedding"), "w") as fout:
			fout.write (f"{list (map(str, overall_var_embedding.tolist()))}\n")

		logging.info (f"Overall statistics for {word} written")

if __name__ == "__main__":
	main (readArgs ())
