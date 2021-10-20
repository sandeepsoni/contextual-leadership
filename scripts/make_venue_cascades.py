""" Create a single file that contains all the relevant data for 
    the rest of the downstream tasks.
"""
import pickle
import pandas as pd
import logging

logging.basicConfig(
	level=logging.INFO, 
	format= '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
	datefmt='%H:%M:%S'
)

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

def main ():
	## All the constants
	innovs_file = "../data/innovs.pkl"
	cascades_file = "../data/cascades.pkl"
	data_file = "../data/data-v1.001.pkl"
	venues_file = "../data/venues.tsv"
	keep_venues_file = "../data/keep_venues.txt"

	num_cascades = 1000
	events_per_cascade=100

	# Read from file
	with open (innovs_file, "rb") as fin:
		innovs = pickle.load (fin) # a list of innovations

	with open (cascades_file, "rb") as fin:
		cascades = pickle.load (fin)

	venues = pd.read_csv (venues_file, sep='\t')
	venue_map = get_venue_mapping (venues)

	with open (keep_venues_file) as fin:
		keep_venues = set ()
		for line in fin:
			keep_venues.add (line.strip())

	logging.info (f'Number of cascades originally: {len(cascades)}')
	cascades = collapse_cascades (cascades, venue_map, keep_venues)
	logging.info (f'Number of cascades after collapsing venues: {len(cascades)}')


	filtered_cascades = limit_cascades (cascades, num_cascades=num_cascades, events_per_cascade=events_per_cascade)

	n_events_before = sum([len(cascade) for cascade in cascades])
	n_events_now = sum([len(cascade) for cascade in filtered_cascades])
	logging.info (f'Number of cascades filtered from {len(cascades)} to {len(filtered_cascades)}')
	logging.info (f'Number of events filtered from {n_events_before} to {n_events_now}')


	idx, iidx = make_channels_map (filtered_cascades)
	remapped_cascades = remap_cascades (filtered_cascades, idx)
	formatted_cascades = format_cascades (remapped_cascades)

	with open (data_file, 'wb') as fout: pickle.dump ((idx, iidx, formatted_cascades, innovs), fout)
	logging.info (f'All relevant data dumped in {data_file}')	

if __name__ == "__main__":
	main ()
