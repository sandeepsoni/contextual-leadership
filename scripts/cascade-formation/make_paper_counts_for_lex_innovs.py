import argparse
import os
from collections import Counter
from tqdm import tqdm

def readArgs ():
	parser = argparse.ArgumentParser (description="Make cascades file")
	parser.add_argument ("--innovs-file", type=str, required=True, help="File contains the semantic innovations")
	parser.add_argument ("--word-embeddings-file", type=str, required=True, help="File contains the embeddings")
	parser.add_argument ("--counts-file", type=str, required=True, help="File contains counts")
	args = parser.parse_args ()
	return args

def main (args):
	# Read the innovations from file
	words = set ()
	with open (args.innovs_file) as fin:
		for line in fin:
			parts = line.strip().split (",")
			words.add (parts[0])

	cascade = list ()
	with open (args.counts_file, "w") as fout, open (args.word_embeddings_file) as fin:
		for i, line in enumerate (tqdm (fin)):
			parts = line.strip().split ("\t")
			paper_id, year, token_position, word = float (parts[0]), int (parts[1]), int (parts[2]), parts[3]
			cascade.append ([word, year, paper_id, token_position])

		c = Counter ([(item[0], item[1], item[2]) for item in cascade])
		for (word, year, paper_id), count in c.items():
			fout.write (f"{word}\t{year}\t{paper_id}\t{count}\n")

if __name__ == "__main__":
	main (readArgs ())
