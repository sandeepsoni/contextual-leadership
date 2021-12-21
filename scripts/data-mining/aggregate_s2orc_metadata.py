import argparse
import glob
import os
import json
import re

def readArgs ():
	parser = argparse.ArgumentParser (description="Find ACL papers in s2orc corpus")
	parser.add_argument ("--acl-json-file", type=str, required=True, help="ACL Bibliography")
	parser.add_argument ("--s2orc-dir", type=str, required=True, help="Directory contains the s2orc corpus")
	parser.add_argument ("--s2orc-json-file", type=str, required=True, help="S2Orc file that contain ACL bib entries")
	args = parser.parse_args ()
	return args

def preprocess_title (title):
	title = title.lower ()
	# remove everything except alphabets, numbers and underscores 
	s = re.sub(r'\W+', '', title)
	return s

def main (args):
	acl_anthology_titles = dict ()
	with open (args.acl_json_file) as fin:
		for line in fin:
			js = json.loads (line)
			short_title = preprocess_title (js["title"])
			if short_title not in acl_anthology_titles:
				acl_anthology_titles[short_title] = list ()
			#acl_anthology_titles[preprocessed_title].append ((js["title"], js["year"]))
			acl_anthology_titles[short_title].append (js)
			

	print (len (acl_anthology_titles))

	# Do something about the duplicates!

	counter = 0
	for key in acl_anthology_titles:
		if len (acl_anthology_titles[key]) > 1:
			#print (key, acl_anthology_titles[key])
			counter += 1

	print (counter)


	# Now lookup these short titles and years in s2orc dataset. Keep only those which have a processed pdf file
	
	matches = 0
	
	with open (args.s2orc_json_file, "w") as fout:
		for filename in glob.glob (os.path.join (args.s2orc_dir, "*.jsonl")):
			with open (filename) as fin:
				for line in fin:
					js = json.loads (line)
					#if (js["title"].lower(), js["year"]) in titles:
					try:
						preproc_title = preprocess_title (js["title"])
						if preproc_title in acl_anthology_titles and js["has_pdf_parse"]:
							years = sorted ([item["year"] for item in acl_anthology_titles[preproc_title]])
							for i, year in enumerate (years):
								if js["year"] == year:
									item = acl_anthology_titles[preproc_title][i]
									js["acl_bib_entry"] = item
									fout.write (f'{json.dumps (js)}\n')
									matches += 1
						#titles.remove ((js["title"].lower(), js["year"]))
						#titles.remove (preprocess_title (js["title"]))
					except KeyError as e:
						print (js)

	print (matches)
	#print (matches, len (titles))
	#print (titles)

if __name__ == "__main__":
	main (readArgs ())
