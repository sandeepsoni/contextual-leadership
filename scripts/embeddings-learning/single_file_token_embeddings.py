import argparse
import json
import logging
from tqdm import tqdm

logging.basicConfig(
	format='%(asctime)s %(levelname)-8s %(message)s',
	level=logging.INFO,
	datefmt='%Y-%m-%d %H:%M:%S')

def readArgs ():
	parser = argparse.ArgumentParser (description="Take multiple files and join them into one after including other fields")
	parser.add_argument ("--filenames", type=str, required=True, nargs="+", help="Files contain token embeddings on a sample")
	parser.add_argument ("--output-file", type=str, required=True, help="File will be output to contain all the embeddings")
	parser.add_argument ("--metadata-file", type=str, required=True, help="The metadata file contains additional fields")
	args = parser.parse_args ()
	return args

def main (args):
	paper_ids = dict ()
	with open (args.metadata_file) as fin:
		for line in fin:
			js = json.loads (line)
			if js["paper_id"] not in paper_ids:
				paper_ids[js["paper_id"]] = js["metadata"]["year"]

	with open (args.output_file, "w") as fout:
		for filename in args.filenames:
			with open (filename) as fin:
				for i, line in enumerate (tqdm (fin)):
					parts = line.strip().split ("\t") # split on tab
					paper_id = parts[0]
					year = paper_ids[paper_id]
					output_string = "\t".join ([paper_id, str(year)] + parts[1:])
					fout.write (f"{output_string}\n")

					if (i+1) % 10000 == 0:
						logging.info (f"{i+1} tokens in {filename} processed")

if __name__ == "__main__":
	main (readArgs ())
