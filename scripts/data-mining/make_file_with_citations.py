import argparse
import json
from collections import Counter

def readArgs ():
	parser = argparse.ArgumentParser (description="Create a file that contains the temporal distribution of citatations")
	parser.add_argument ("--src-file", type=str, required=True, help="File contains the overall data")
	parser.add_argument ("--tgt-file", type=str, required=True, help="File contains citation distribution per paper")
	args = parser.parse_args ()
	return args

def main (args):
	publication_years = dict ()
	with open (args.src_file) as fin:
		for line in fin:
			js = json.loads (line)
			paper_id = js["paper_id"]
			year = js["metadata"]["year"]
			publication_years[paper_id] = year

	citation_dist = dict ()
	with open (args.src_file) as fin, open (args.tgt_file, "w") as fout:
		for line in fin:
			js = json.loads (line)
			paper_id = js["paper_id"] #paper_id
			year = js["metadata"]["year"] #year
			inlinks = js["inlinks"] # get all inlinks
			citation_dist = dict (Counter ([int (publication_years[inlink]) for inlink in inlinks])) # yearly distribution of citations	
			out_js = {}
			out_js["paper_id"] = paper_id
			out_js["publication_year"] = year
			out_js["n_citations"] = len (inlinks)
			out_js["citation_distribution"] = citation_dist
			fout.write (f"{json.dumps (out_js)}\n")
	

if __name__ == "__main__":
	main (readArgs ())
