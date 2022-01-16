""" Create a single file that contains all the relevant data for 
    the rest of the downstream tasks.
"""
import pickle
import pandas as pd
import logging
import os
import sys
import argparse
if not os.path.abspath ("../../modules") in sys.path:
	sys.path.append (os.path.abspath ("../../modules"))
import hpio

logging.basicConfig(
	level=logging.INFO, 
	format= '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
	datefmt='%H:%M:%S'
)

def readArgs ():
	parser = argparse.ArgumentParser (description="Create a fast loadable dataset from file")
	parser.add_argument ("--input-cascades-file", type=str, required=True, help="File contains per word cascade of events")
	parser.add_argument ("--output-pickle-file", type=str, required=True, help="File contains all cascades")
	parser.add_argument ("--num-cascades", type=int, required=False, default=5000, help="Number of cascades to consider")
	parser.add_argument ("--events-per-cascades", type=int, required=False, default=100, help="Number of events per cascade")
	args = parser.parser_args ()
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

def main (args):
	logging.info (f"Read all the data from file {args.input_cascades_file}")
	# Read all cascade data from file
	idx, iidx, cascades, innovs = hpio.read_cascades_from_file (args.input_cascades_file)	
	logging.info (f'Number of cascades originally: {len(cascades)}')

	filtered_cascades = limit_cascades (cascades, num_cascades=args.num_cascades, events_per_cascade=args.events_per_cascade)

	n_events_before = sum([len(cascade) for cascade in cascades])
	n_events_now = sum([len(cascade) for cascade in filtered_cascades])
	logging.info (f'Number of cascades filtered from {len(cascades)} to {len(filtered_cascades)}')
	logging.info (f'Number of events filtered from {n_events_before} to {n_events_now}')

	return

	idx, iidx = make_channels_map (filtered_cascades)
	remapped_cascades = remap_cascades (filtered_cascades, idx)
	formatted_cascades = format_cascades (remapped_cascades)

	with open (data_file, 'wb') as fout: pickle.dump ((idx, iidx, formatted_cascades, innovs), fout)
	logging.info (f'All relevant data dumped in {data_file}')	

if __name__ == "__main__":
	main (readArgs ())
