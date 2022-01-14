import argparse
from tqdm import tqdm

def readArgs ():
	parser = argparse.ArgumentParser (description="Create a list of semantic changes")
	parser.add_argument ("--computed-scores-file", type=str, required=True, help="File contains computed scores")
	parser.add_argument ("--keep-words-file", type=str, required=True, help="File contains words that should be kept")
	parser.add_argument ("--top-words", type=int, required=False, default=5000, help="Filter from these many top words")
	parser.add_argument ("--final-words-file", type=str, required=True, help="File contains final semantic changes")
	args = parser.parse_args ()
	return args

def main (args):
	# Read the vocabulary from file
	vocab = set ()
	with open (args.computed_scores_file) as fin:
		for i, line in enumerate (tqdm (fin)):
			if i < args.top_words:
				vocab.add (line.strip().split ("\t")[0])

	# Read the words to keep
	keep_words = set ()
	with open (args.keep_words_file) as fin:
		for line in tqdm (fin):
			keep_words.add (line.strip())

	words = vocab & keep_words
	with open (args.final_words_file, "w") as fout: 
		for word in words:
			fout.write (f"{word}\n")

if __name__ == "__main__":
	main (readArgs ())
