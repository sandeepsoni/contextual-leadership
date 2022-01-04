import argparse
import os
import logging
from tqdm import tqdm

logging.basicConfig(
	format='%(asctime)s %(levelname)-8s %(message)s',
	level=logging.INFO,
	datefmt='%Y-%m-%d %H:%M:%S')

def readArgs ():
	parser = argparse.ArgumentParser (description="Split the files by word and year")
	parser.add_argument ("--words-file", type=str, required=True, help="File contains candidate words")
	parser.add_argument ("--embeddings-word-dir", type=str, required=True, help="Directory contains subdirectories per candidate word")
	parser.add_argument ("--embeddings-file", type=str, required=True, help="File contains the contextual embeddings per word")
	args = parser.parse_args ()
	return args

def main (args):
	words = set ()
	with open (args.words_file) as fin:
		for line in fin:
			words.add (line.strip())

	# Create the directory
	os.makedirs (args.embeddings_word_dir, exist_ok=True)

	# Create the subdirectories
	for word in words:
		os.makedirs (os.path.join (args.embeddings_word_dir, word), exist_ok=True)

	# Now iterate over token at a time
	with open (args.embeddings_file) as fin:
		for line_no, line in enumerate (tqdm(fin)):
			parts = line.strip().split("\t")
			word, year = parts[3], parts[1]
			if word in words: # do this only if it is one of the candidates
				word_subdir = os.path.join (args.embeddings_word_dir, word)
				with open (os.path.join (word_subdir, f"{year}.tsv"), "a") as fout:
					fout.write (line)

			if ((line_no+1) % 100000) == 0:
				logging.info (f"Processed {line_no+1} lines from {args.embeddings_file}")	

if __name__ == "__main__":
	main (readArgs ())
