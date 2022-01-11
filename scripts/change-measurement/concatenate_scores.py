import argparse
import os
import glob
from tqdm import tqdm

def readArgs ():
	parser = argparse.ArgumentParser (description="Concatenate the scores for all the words")
	parser.add_argument ("--wordfiles-dir", required=True, type=str, help="Directory contains all the words in different file chunks")
	parser.add_argument ("--word-embeddings-dir", required=True, type=str, help="Directory contains word embeddings and their statistics")
	parser.add_argument ("--scores-suffix", required=False, type=str, default="frequency_accounting.max_score", help="Suffix for the scores file")
	parser.add_argument ("--changepoints-file", required=True, type=str, help="File contains transition points of words and their scores")
	args = parser.parse_args ()
	return args

def main (args):
	words = set ()
	for filename in glob.glob (os.path.join (args.wordfiles_dir, "*.keepparts")):
		with open (filename) as fin:
			for line in fin:
				words.add (line.strip())

	scores = dict ()
	for word in words:
		scores_file = os.path.join (args.word_embeddings_dir, word, f"{word}.{args.scores_suffix}")
		with open (scores_file) as fin:
			parts = fin.read ().strip().split ("\t")
			parts_string = "\t".join ([word] + parts)
			scores[word] = parts_string

	with open (args.changepoints_file, "w") as fout:
		for _word, _parts_string in sorted (scores.items(), key=lambda x:float (x[1].split("\t")[2]), reverse=True):
			fout.write (f"{_parts_string}\n")

if __name__ == "__main__":
	main (readArgs ())
