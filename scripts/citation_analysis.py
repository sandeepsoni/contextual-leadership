import click
import pickle
import logging
from collections import Counter
import pandas as pd

def filter_citations (venue_map, citations):
	filtered_citations = list ()
	for src, tgt in citations:
		if src in venue_map and tgt in venue_map:
			filtered_citations.append ((src, tgt))

	return filtered_citations

def make_df (idx, iidx, hp_params, citations):
	filtered_citations = filter_citations (idx, citations)

	incoming_citations = {}
	for _, tgt in filtered_citations:
		if tgt not in incoming_citations:
			incoming_citations[tgt] = 0
		incoming_citations[tgt] += 1

	outgoing_citations = {}
	for src, _ in filtered_citations:
		if src not in outgoing_citations:
			outgoing_citations[src] = 0
		outgoing_citations[src] += 1

	mu, b, c, s = hp_params

	keys = [key for key in idx] # keys
	trans_params = [b[idx[key]] for key in keys]
	base_params = [mu[idx[key]] for key in keys]
	recep_params = [c[idx[key]] for key in keys]
	se_params = [s[idx[key]] for key in keys]
	in_cites = [incoming_citations[key] for key in keys]
	out_cites = [outgoing_citations[key] for key in keys]
	
	df = pd.DataFrame ({"venues": keys, 
						"sem_transmissibility": trans_params, 
						"sem_receptiveness": recep_params, 
						"sem_baseline": base_params, 
						"sem_selfexcitation": se_params, 
						"incoming_citations": in_cites, 
						"outgoing_citations": out_cites})
	return df

def remap_citations (citations, venue_map, keep_venues):
	new_citations = list ()
	for src, tgt in citations:
		if src in venue_map and tgt in venue_map and src in keep_venues and tgt in keep_venues:
			new_citations.append ((venue_map[src], venue_map[tgt]))

	return new_citations

@click.command ()
@click.option ('--data-file', default='../data/data-v1.001.pkl', help='File contains the cascades and index maps')
@click.option ('--params-file', default='../data/params-v1.001.pkl', help='File contains the parameters of the coarse HP model')
@click.option ('--citations-file', default='../data/venue_citations.pkl', help='File contains the venue citations')
@click.option ('--venues-file', default='../data/venues.tsv', help='File contains the venue information')
@click.option ('--keep-venues-file', default='../data/keep_venues.txt', help='Only keep these venues')
@click.option ('--output-file', default='../data/inf-v1.001.csv', help='File contains citation and influence relations')
@click.option ('--log-file', default='citation_analysis.log', help='File contains the log messages for citation analysis')
def main (data_file, params_file, citations_file, venues_file, keep_venues_file, output_file, log_file):
	logging.basicConfig(
		filename=log_file,
		level=logging.INFO,
		format= '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
		datefmt='%H:%M:%S'
	)
	logging.info ('Script execution...started')
	# Read the innovations
	with open (data_file, 'rb') as fin:
		idx_channels, iidx_channels, cascades, innovs = pickle.load (fin)

	logging.info ('Data...loaded')

	# Read the params
	with open (params_file, 'rb') as fin:
		coarse_hp_params = pickle.load (fin)

	logging.info ('HP params...loaded')
	
	# Read the citations
	with open (citations_file, 'rb') as fin:
		citations = pickle.load (fin)

	logging.info ('Citations ... loaded')

	# Since we collapsed the workshops into one placeholder,
	# it's best to run one more pass on the citations, 
	# replacing the specific workshops with the placeholder
	venues = pd.read_csv (venues_file, sep='\t')

	venue_map = dict ()
	for i, row in venues.iterrows():
		assigned_venue = row['assigned_venues']
		if 'ws' in eval(row['venues']):
			venue_map[assigned_venue] = 'ws'
		else:
			venue_map[assigned_venue] = assigned_venue

	with open (keep_venues_file) as fin:
		keep_venues = set ()
		for line in fin:
			keep_venues.add (line.strip())

	new_citations = remap_citations (citations, venue_map, keep_venues)
	logging.info (f'Number of citations before: {len(citations)}, number of citations later: {len (new_citations)}')

	df = make_df (idx_channels, iidx_channels, coarse_hp_params, new_citations)
	logging.info ('Influence relations...calcualated')

	df.to_csv (output_file, sep='\t', index=False, header=True)
	logging.info ('Influence relations...written')

if __name__ == "__main__":
	main ()
