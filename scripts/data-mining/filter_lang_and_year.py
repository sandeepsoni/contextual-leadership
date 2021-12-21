import argparse
import json
import os
import langid
import logging


logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', 
					level=logging.INFO, 
					datefmt='%Y-%m-%d %H:%M:%S')

def readArgs ():
	parser = argparse.ArgumentParser (description="Get lang ids")
	parser.add_argument ("--input-filename", required=True, type=str, help="JSONL file contains input text")
	parser.add_argument ("--output-filename", required=True, type=str, help="JSONL file contains language id and confidence")
	parser.add_argument ("--lang", required=False, type=str, default="en", help="Filter to keep only English language papers")
	parser.add_argument ("--from-year", required=False, type=int, default=1990, help="Keep only papers that are on or after this year")
	parser.add_argument ("--to-year", required=False, type=int, default=2019, help="Keep only the paper that are on or before this year")
	args = parser.parse_args ()
	return args

def main (args):
	paper_ids = set ()
	with open (args.input_filename) as fin, open (args.output_filename, "w") as fout:
		for i, line in enumerate (fin):
			js = json.loads (line)
			paper_id = js["paper_id"]
			year = js["metadata"]["year"]
			if paper_id not in paper_ids and year >= args.from_year and year <= args.to_year:
				text = " ".join ([item["text"] for item in js["abstract"]])
				lang, score = langid.classify(text)
				if lang == args.lang:
					js["lang_id"] = lang
					fout.write (f"{json.dumps (js)}\n")
					if (i+1) % 1000 == 0:
						logging.info  (f'In file {args.input_filename}, {i+1} records fully processed')
					paper_ids.add (paper_id)

if __name__ == "__main__":
	main (readArgs ())
