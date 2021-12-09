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
	parser.add_argument ("--data-dir", type=str, required=True, help="Data directory contains JSONL files")
	parser.add_argument ("--sleep-time", type=int, required=False, default=30, help="The sleep time")
	parser.add_argument ("--output-file", type=str, required=True, help="File contains the url and the extracted venues")
	args = parser.parse_args ()
	return args

def getACLIds (dirname):
	ids = list ()
	for filename in glob.glob (os.path.join(dirname, '*.jsonl')):
		with open (filename) as fin:
			for line in fin:
				js = ujson.loads (line)
				acl_id = js['acl_id']
				ids.append (acl_id)
	return ids

def ids2url (acl_ids):
	prefix = 'https://www.aclweb.org/anthology/'
	return [prefix + acl_id for acl_id in acl_ids]

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
	acl_ids = getACLIds (args.data_dir)
	urls = ids2url (acl_ids)
	all_venues = list ()
	for i, url in enumerate (urls):
		venue = extractVenues (url)
		all_venues.append (venue)		
		if (i+1) % 100 == 0:
			logging.info (f'Record {i+1}: {acl_ids[i]}, {urls[i]}, {venue}')
			sleep (args.sleep_time)

	df = pd.DataFrame ({'acl_ids': acl_ids, 'urls': urls, 'venues': all_venues})
	df.to_csv (args.output_file, sep='\t', header=True, index=False)

if __name__ == "__main__":
	main (readArgs ())
