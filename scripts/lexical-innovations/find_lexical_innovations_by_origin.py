import argparse
import json
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

def readArgs ():
	parser = argparse.ArgumentParser (description="Find lexical innovations by origin")
	parser.add_argument ("--input-filename", type=str, required=True, help="File contains the text")
	parser.add_argument ("--field-name", type=str, required=False, default="abstract", help="field name that contains the text")
	parser.add_argument ("--from-year", type=int, required=False, default=1990, help="The start year")
	parser.add_argument ("--till-year", type=int, required=False, default=2019, help="The end year")
	parser.add_argument ("--counts-threshold", type=int, required=False, default=10, help="The frequency threshold for filtering out words")
	parser.add_argument ("--output-filename", type=str, required=True, help="File will contain the lexical innovations")
	args = parser.parse_args ()
	return args

def get_processed_text_as_tokens (text_json):
	all_tokens = list ()
	for item in text_json:
		text = item["text"]
		cite_spans = [cite_item["text"] for cite_item in item["cite_spans"]]
		for cite_span in cite_spans:
			text = text.replace (cite_span, " ")
		ref_spans = [ref_item["text"] for ref_item in item["ref_spans"]]
		for ref_span in ref_spans:
			text = text.replace (ref_span, " ")
		tokens = text.lower().split ()
		for token in tokens:
			if token.isalpha ():
				all_tokens.append (token)

	return all_tokens

def main (args):
	already_seen = set ()
	first_seen = dict ()
	word_counts = defaultdict (int)
	with open (args.input_filename) as fin:
		for line in tqdm (fin):
			js = json.loads (line)
			tokens = get_processed_text_as_tokens (js[args.field_name])
			#tokens = [token.lower() for token in js[args.field_name] if token.isalpha()]
			year = js["metadata"]["year"]
			for token in tokens:
				token = token.strip()
				if token not in already_seen:
					first_seen[token] = year
					already_seen.add (token)
				else:
					first_seen[token] = min (year, first_seen[token])
				word_counts[token] += 1

	rows = [[key,first_seen[key]] for key in first_seen if word_counts[key] >= args.counts_threshold and first_seen[key] >= args.from_year and first_seen[key] <= args.till_year]
	df = pd.DataFrame (rows, columns=["Word", "Year"])
	df = df.sort_values (by="Year")
	df.to_csv (args.output_filename, sep=",", header=True, index=False)
	

if __name__ == "__main__":
	main (readArgs ())
