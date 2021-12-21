import argparse
import pandas as pd
import glob
import os
import logging
import ujson
import urllib.request
from bs4 import BeautifulSoup
from time import sleep

logging.basicConfig(
	format='%(asctime)s %(levelname)-8s %(message)s',
	level=logging.INFO,
	datefmt='%Y-%m-%d %H:%M:%S')

def readArgs ():
	parser = argparse.ArgumentParser (description='Scrape the ACL anthology to extract venues for each paper')
	parser.add_argument ("--jsonl-file", type=str, required=True, help="File contains input")
	parser.add_argument ("--venues-file", type=str, required=True, help="File contains output")
	parser.add_argument ("--sleep-time", type=int, required=False, default=30, help="The sleep time")
	args = parser.parse_args ()
	return args

def extractVenues (url):
	venues = list ()
	try:
		with urllib.request.urlopen(url) as response:
			html = response.read()
			soup = BeautifulSoup(html, 'html.parser')
			for elem in soup.findAll ('a', href=True):
				href = elem['href']
				if 'venues' in href:
					venues.append (href.strip('/').split('/')[-1])
	except Exception as e:
		pass
	return venues

def main (args):
	with open (args.jsonl_file) as fin, open (args.venues_file, "w") as fout:
		for i, line in enumerate (fin):
			js = ujson.loads (line)
			url = js["metadata"]["acl_bib_entry"]["url"]
			if url.startswith ("https://aclanthology.org"):
				venue = extractVenues (url)
				fout.write (f"{url}\t{venue}\n")
				if (i+1) % 100 == 0:
					logging.info (f'Record {i+1}: {url}, {venue}')
					sleep (args.sleep_time)
			
if __name__ == "__main__":
	main (readArgs ())
