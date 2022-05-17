import argparse
from collections import defaultdict
import json
from tqdm import tqdm
import pandas as pd

def readArgs ():
	parser = argparse.ArgumentParser (description="Create a list of lexical innovations by frequency")
	parser.add_argument ("--input-filename", required=True, type=str, help="File contains the text")
	parser.add_argument ("--output-filename", required=True, type=str, help="File contains lexical innovations")
	parser.add_argument ("--counts-threshold", required=False, type=int, default=10, help="counts threshold ")
	parser.add_argument ("--from-year", type=int, required=False, default=1990, help="from year")
	parser.add_argument ("--till-year", type=int, required=False, default=2019, help="till year")
	args = parser.parse_args ()
	return args	

def main (args):
	# Get the count distribution of each word in the vocabulary
	words = {year: defaultdict (int) for year in range (args.from_year, args.till_year+1)}
	with open (args.input_filename) as fin:
		for line in tqdm(fin):
			js = json.loads (line)
			tokens = js["tokenized_text"]
			year = js["metadata"]["year"]
			for token in tokens:
				token = token.lower().strip()
				words[year][token] += 1

	total_counts = {year: sum(words[year].values()) for year in range (args.from_year, args.till_year+1)}

	freq_differences = {year: dict () for year in range (args.from_year+1, args.till_year+1)}
	for segmentation_year in range (args.from_year+1, args.till_year+1):
		common_vocab = set ([word for word in words[segmentation_year-1]]) & set ([word for word in words[segmentation_year]])
		common_vocab = [w for w in common_vocab if w.isalpha() and (words[segmentation_year - 1][w] >= args.counts_threshold and words[segmentation_year][w] >= args.counts_threshold)]
		for w in common_vocab:
			freq_differences[segmentation_year][w] = words[segmentation_year][w]/words[segmentation_year-1][w]
	
	rows  = list ()
	for y in freq_differences:
		for w in freq_differences[y]:
			rows.append ([y, w, freq_differences[y][w], words[y-1][w], words[y][w]])

	df = pd.DataFrame (rows, columns=["year", "word", "frequency_ratio", "count_before", "count_after"])
	groups = df.groupby ("word")
	max_df = pd.concat([group.sort_values (by="frequency_ratio", ascending=False).head(1) for name, group in groups])
	max_df = max_df.reset_index()
	max_df = max_df.sort_values (by="frequency_ratio", ascending=False)
	max_df.to_csv (args.output_filename, sep=",", header=True, index=False)
			

if __name__ == "__main__":
	main (readArgs ())
