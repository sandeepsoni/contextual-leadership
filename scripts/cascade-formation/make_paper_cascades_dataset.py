""" Create a single file that contains all the relevant data for 
    the rest of the downstream tasks.
"""
import pickle
import pandas as pd
import logging
import os
import sys
import argparse
import json
import pandas as pd

logging.basicConfig(
	level=logging.INFO, 
	format= '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
	datefmt='%H:%M:%S'
)

def readArgs ():
	parser = argparse.ArgumentParser (description="Create a fast loadable dataset from file")
	parser.add_argument ("--input-cascades-file", type=str, required=True, help="File contains per word cascade of events")
	parser.add_argument ("--output-counts-file", type=str, required=True, help="File contains counts as JSON per line")
	args = parser.parse_args ()
	return args	

def limit_cascades (cascades, num_cascades=1000, events_per_cascade=100):
	new_cascades = [[item for item in cascade[0:events_per_cascade]] for cascade in cascades[0:num_cascades]]
	return new_cascades

def make_channels_map (cascades):
	idx = dict ()
	for cascade in cascades:
		for channel, _ in cascade:
			if channel not in idx:
				idx[channel] = len (idx)

	iidx = {idx[channel]: channel for channel in idx}
	return idx, iidx

def remap_cascades (cascades, idx):
	new_cascades = [[(idx[channel], time) for channel, time in cascade] for cascade in cascades]
	return new_cascades

def transform_cascade (cascade):
	new_cascades = {ts:[] for ch, ts in cascade}
	for channel, timestamp in cascade:
		new_cascades[timestamp].append (channel)

	return new_cascades
	
def format_cascades (cascades):
	return [transform_cascade (cascade) for cascade in cascades]

def get_venue_mapping (venues):
	maps = dict ()
	for i, row in venues.iterrows():
		assigned_venue = row['assigned_venues']
		if 'ws' in eval(row['venues']):
			maps[assigned_venue] = 'ws'
		else:
			maps[assigned_venue] = assigned_venue	

	return maps

def collapse_cascades (cascades, venue_map, keep_venues):
	new_cascades = list ()
	for cascade in cascades:
		new_cascade = list ()
		for channel, timestamp in cascade:
			channel_name = venue_map[channel]
			if channel_name in keep_venues:
				new_cascade.append ((channel_name, timestamp))
		
		if len (new_cascade) > 0:
			new_cascades.append (new_cascade)

	return new_cascades

def create_record (word, paper_id, year, num_innovations, history):
	js = dict ()
	js["word"] = word
	js["year"] = year
	js["paper_id"] = paper_id
	js["num_innovations"] = num_innovations
	js["previous_papers"] = list ()
	for key, value in history.items():
		js["previous_papers"].append ({"paper_id": key,
									   "year": value})

	return js

def get_distribution (counts_df):
	dist = dict ()
	for row_num, row in counts_df.iterrows():
		if row["word"] not in dist:
			dist[row["word"]] = dict ()
		if row["year"] not in dist[row["word"]]:
			dist[row["word"]][row["year"]] = list ()
		dist[row["word"]][row["year"]].append ((row["paper_id"], row["num_innovations"]))

	return dist

def main (args):
	logging.info (f"Read all the data from file {args.input_cascades_file}")
	# Read all cascade data from file
	paper_counts = pd.read_csv(args.input_cascades_file, sep="\t", names=["word", "year", "paper_id", "num_innovations"])	
	publication_years = {row["paper_id"]: row["year"] for _, row in paper_counts.iterrows()}
	word_usage_distribution = get_distribution (paper_counts)

	with open (args.output_counts_file, "w") as fout:
		for row_num, row in paper_counts.iterrows():
			year = row["year"]
			word = row["word"] 
			paper_id = row["paper_id"]
			num_innovations = row["num_innovations"]

			history = [word_usage_distribution[word][y] for y in word_usage_distribution[word] if y < year]
			history = {paper_id: publication_years[paper_id] for items in history for paper_id, _ in items}

			record = create_record (word, paper_id, year, num_innovations, history)
			fout.write (f"{json.dumps (record)}\n")

	logging.info (f'All relevant data dumped in {args.output_counts_file}')

if __name__ == "__main__":
	main (readArgs ())
