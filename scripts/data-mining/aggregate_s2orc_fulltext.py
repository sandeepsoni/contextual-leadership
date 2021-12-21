import argparse
import glob
import os
import json

def readArgs ():
	parser = argparse.ArgumentParser (description="Get the full text of the papers")
	parser.add_argument ("--input-filename", type=str, required=True, help="File contains the metadata")
	parser.add_argument ("--pdf-parses-dir", type=str, required=True, help="Directory contains the parses")
	parser.add_argument ("--output-filename", type=str, required=True, help="File contains metadata and full text")
	args = parser.parse_args ()
	return args

def main (args):
	metadata = dict ()
	with open (args.input_filename) as fin:
		for line in fin:
			js = json.loads (line)
			metadata[js["paper_id"]] = js

	with open (args.output_filename, "w") as fout:
		for filename in glob.glob (os.path.join (args.pdf_parses_dir, "*.jsonl")):
			with open (filename) as fin:
				for line in fin:
					js = json.loads (line)
					if js["paper_id"] in metadata:
						if len (js["abstract"]) > 0 and len (js["body_text"]) > 0:
							js["metadata"] = metadata[js["paper_id"]]
							fout.write (f"{json.dumps (js)}\n")	

if __name__ == "__main__":
	main (readArgs ())
