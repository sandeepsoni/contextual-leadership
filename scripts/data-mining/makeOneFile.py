import argparse
import json
import langid
import os

def readArgs ():
	parser = argparse.ArgumentParser (description="Create a single file representation of the ACL papers in semantic scholar")
	parser.add_argument ('--input-file', type=str, required=True, help="File contains all the abstracts of the ACL papers")
	parser.add_argument ('--output-file', type=str, required=True, help="File contains everything put together")
	args = parser.parse_args ()
	return args

def scrub_ref_spans (item):
	for span_item in item["ref_spans"]:
		ref_id = span_item["ref_id"]
		ref_span = span_item["text"]
		if ref_id.startswith ("FIG"):
			item["text"] = item["text"].replace (ref_span, "Figure")
		elif ref_id.startswith ("TAB"):
			item["text"] = item["text"].replace (ref_span, "Table")
		else:
			item["text"] = item["text"].replace (ref_span, " ")

	return item

def scrub_cite_spans (item):
	for span_item in item["cite_spans"]:
		ref_id = span_item["ref_id"]
		ref_span = span_item["text"]
		if ref_id.startswith ("BIB"):
			item["text"] = item["text"].replace (ref_span, " ")

	return item
	
def main (args):
	### Read the input file ###
	paper_ids = set ()
	with open (args.input_file) as fin:
		for line in fin:
			js = json.loads (line.strip())
			paper_ids.add (js['paper_id'])

	### Read the input file and create a citation network ###	
	outlinks = dict ()
	inlinks = dict ()
	with open (args.input_file) as input_file:
		for line in input_file:
			js = json.loads (line.strip())
			paper_id = js["paper_id"]
			inlinks[paper_id] = [citation for citation in set (js["metadata"]["inbound_citations"]) if citation in paper_ids]
			outlinks[paper_id] = [citation for citation in set (js["metadata"]["outbound_citations"]) if citation in paper_ids]	
	########################################################

	### Create JSONs and write to file ###

	done_papers = set ()
	with open (args.output_file, "w") as output_file, open (args.input_file) as input_file:
		for line in input_file:
			record = json.loads (line)
			ID = record["paper_id"]
			if ID in paper_ids and ID not in done_papers:
				items = [scrub_ref_spans (item) for item in record["body_text"]]
				items = [scrub_cite_spans (item) for item in items]
				record["tokenized_text"] = [token for item in items for token in item["text"].split ()]
				record["full_text"] = " ".join (record["tokenized_text"])
				#record["tokens"] = [token for token in record["body_text"].lower().split () if token.isalpha()]
				record["outlinks"] = outlinks.get (ID, [])
				record["inlinks"] = inlinks.get (ID, [])
				done_papers.add (ID)
				output_file.write (f'{json.dumps (record)}\n')
	######################################	

if __name__ == "__main__":
	main (readArgs ())
