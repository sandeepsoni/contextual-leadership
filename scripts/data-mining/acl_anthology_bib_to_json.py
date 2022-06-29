""" Converts a bib file from the ACL anthology into a JSON file. 

* santosh-etal-2020-detecting: Fix unbalanced parenthesis.

* ferret-2014-compounds: Replace the braces with parenthesis in the abstract.

"""

import argparse
from pybtex.database.input import bibtex
import json

def readArgs ():
	parser = argparse.ArgumentParser (description="Read the bib file and convert into a JSON file")
	parser.add_argument ("--bib-file", type=str, required=True, help="bib file")
	parser.add_argument ("--json-file", type=str, required=True, help="JSON file")
	args = parser.parse_args ()
	return args

def main (args):
	parser = bibtex.Parser()
	bib_data = parser.parse_file(args.bib_file)
	keys = list (bib_data.entries.keys ())
	non_venue_keys = list ()
	unparseable_keys = list ()
	with open (args.json_file, "w") as fout:
		for key in keys:
			try:
				js = {}
				js["entry_type"] = bib_data.entries[key].type
				js["title"] = bib_data.entries[key].fields["title"]
				js["year"] = int (bib_data.entries[key].fields["year"])
				if "booktitle" in bib_data.entries[key].fields:
					js["venue"] = bib_data.entries[key].fields["booktitle"]
				elif "journal" in bib_data.entries[key].fields:
					js["venue"] = bib_data.entries[key].fields["journal"]
				else:
					js["venue"] = None
					non_venue_keys.append (key)
				js["url"] = bib_data.entries[key].fields["url"]
				fout.write (f'{json.dumps (js)}\n')
			except Exception as e:
				unparseable_keys.append (key)

if __name__ == "__main__":
	main (readArgs ())
