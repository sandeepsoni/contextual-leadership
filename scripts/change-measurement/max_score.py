import argparse
import os
import tqdm

def readArgs ():
	parser = argparse.ArgumentParser (description="Get the peak score for each word")
	parser.add_argument ("--words-file", required=True, type=str, help="File contains a set of words")
	parser.add_argument ("--word-embeddings-dir", required=True, type=str, help="Directory contains subdirectories for each word")
	args = parser.parse_args ()
	return args

def main (args):
	words = set ()
	with open (args.words_file) as fin:
		for line in fin:
			words.add (line.strip())

	for word in tqdm(words):
		filename = os.path.join (args.word_embeddings_dir, word, f"{word}.computed_scores")
		with open (filename) as fin:
			parts = [line.strip().split("\t") for line in fin]
			scores = [(int(part[1]), float (part[2])) for part in parts]
			max_score = max (scores, key=lambda x:x[1])
		
		output_filename = os.path.join (args.word_embeddings_dir, word, f"{word}.frequency_accounting.max_score")
		with open (output_filename, "w") as fout:
			fout.write (f"{max_score[0]}\t{max_score[1]}\n")
				

if __name__ == "__main__":
	main (readArgs ())
