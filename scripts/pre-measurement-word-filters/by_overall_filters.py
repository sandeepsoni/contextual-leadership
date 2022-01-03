"""
Combine all the filters. We keep words that:

- occur at least 30 times
- are alphabetic
- appear in atmost 90% of the papers (helps remove stop words)
- appear at least 25% of the years

"""

import argparse
import glob
import os

def readArgs ():
	parser = argparse.ArgumentParser (description="Combine all the filters")
	parser.add_argument ("--dirname", type=str, required=True, help="directory name containing all the keep and discard files")
	args = parser.parse_args ()
	return args

def main (args):
	keep_words = list ()
	keepfiles = [filename for filename in glob.glob (os.path.join (args.dirname, "*.keep"))]
	for filename in keepfiles:
		keep_by_type = set ()
		with open (filename) as fin:
			for line in fin:
				word = line.strip()
				keep_by_type.add (word)

		keep_words.append (keep_by_type)

	overall_keepwords = keep_words[0].intersection(*keep_words)
	
	all_files = [filename for filename in glob.glob (os.path.join (args.dirname, "*.keep"))]
	all_files.extend ([filename for filename in glob.glob (os.path.join (args.dirname, "*.discard"))])

	discard_words = set ()
	for filename in all_files:
		with open (filename) as fin:
			for line in fin:
				word = line.strip()
				if word not in overall_keepwords:
					discard_words.add (word)

	with open (os.path.join (args.dirname, "by_overall_filters.keep"), "w") as fout:
		for word in sorted (overall_keepwords):
			fout.write (f"{word}\n")

	with open (os.path.join (args.dirname, "by_overall_filters.discard"), "w") as fout:
		for word in sorted (discard_words):
			fout.write (f"{word}\n")
	
			

if __name__ == "__main__":
	main (readArgs ())
