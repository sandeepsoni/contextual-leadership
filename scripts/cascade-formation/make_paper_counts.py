import argparse
import os
from collections import Counter

def readArgs ():
	parser = argparse.ArgumentParser (description="Make cascades file")
	parser.add_argument ("--innovs-file", type=str, required=True, help="File contains the semantic innovations")
	parser.add_argument ("--word-embeddings-dir", type=str, required=True, help="Directory contains word embeddings and other statistics")
	parser.add_argument ("--confidence-threshold", type=float, required=False, default=0.9, help="Keep only the events that the classifier is confident about")
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

	with open (args.counts_file, "w") as fout:
		for word in words:
			filename = os.path.join (args.word_embeddings_dir, word, f"{word}.classification.tsv")
			cascade = list ()
			with open (filename) as fin:
				for i, line in enumerate (fin):
					if i > 0:
						parts = line.strip().split ("\t")
						prob_true, predicted_label, year, paper_id, token_position  = float (parts[1]), parts[3] == "True", int (parts[4]), parts[5], int (parts[6])
						if predicted_label and prob_true >= args.confidence_threshold:
							cascade.append ([word, year, paper_id, token_position])

			c = Counter ([(item[1], item[2]) for item in cascade])
			for (year, paper_id), count in c.items():
				fout.write (f"{word}\t{year}\t{paper_id}\t{count}\n")

if __name__ == "__main__":
	main (readArgs ())
