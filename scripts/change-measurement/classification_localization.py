import argparse
import json
import glob
import os
import sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def readArgs ():
	parser = argparse.ArgumentParser (description="Classify each instance of a word")
	parser.add_argument ("--groundtruth-file", type=str, required=True, help="File contains the groundtruth i.e. transition year for each change")
	parser.add_argument ("--words-file", type=str, required=True, help="File contains list of words")
	parser.add_argument ("--word-embeddings-dir", type=str, required=True, help="Directory contains embeddings")
	parser.add_argument ("--from-year", type=int, required=False, default=1990, help="The starting year")
	parser.add_argument ("--till-year", type=int, required=False, default=2019, help="The end year")
	parser.add_argument ("--from-index", type=int, required=True, help="The starting index in the list of words")
	parser.add_argument ("--till-index", type=int, required=True, help="The ending index in the list of words")
	args = parser.parse_args ()
	return args

def standardize (X):
	return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

def balanced_random_sample (X, y, samples=100):
	ind1 = np.random.choice (np.where (~y)[0], samples)
	ind2 = np.random.choice (np.where (y)[0], samples)
	ind = np.concatenate ((ind1, ind2))
	ind = np.random.permutation (ind)
	ys = y[ind]
	Xs = X[ind]
	return ys, Xs

def make_per_instance_frame (probs, true_labels, predicted_labels, paper_ids, positions):
	d = {'probFalse': probs[:,0], 
		 'probTrue': probs[:,1],
		 'true_labels': true_labels,
		 'predicted_labels': predicted_labels,
		 'paper_id': paper_ids,
		 'token_position': positions}
	return pd.DataFrame (d)

def make_meta_JSON (word, year, true_labels, predicted_labels):
	js = {}
	js['word'] = word
	js['year'] = year
	js['accuracy'] = accuracy_score (true_labels, predicted_labels)
	js['precision'] = precision_score (true_labels, predicted_labels)
	js['recall'] = recall_score (true_labels, predicted_labels)
	js['auc'] = roc_auc_score (true_labels, predicted_labels)
	js['classification_report'] = classification_report (true_labels, predicted_labels)
	return js

def read_stats_from_file (filename, sep='\t'):
	embeddings = list ()
	paper_ids = list ()
	positions = list ()
	with open (filename) as fin:
		for line in fin:
			parts = line.strip().split (sep)
			paper_ids.append (parts[0])
			positions.append (parts[2])
			embeds = list(map(float, parts[-1].split ()))
			embeddings.append (embeds)

	return paper_ids, positions, embeddings

def read_groundtruth_file (filename, sep="\t"):
	groundtruth = dict ()
	with open (filename) as fin:
		for line in fin:
			parts = line.strip().split (sep)
			groundtruth[parts[0]] = int (parts[1])

	return groundtruth

def main (args):
	# Read the groundtruth from file
	groundtruth = read_groundtruth_file (args.groundtruth_file)

	# Read words one at a time
	words = list ()
	with open (args.words_file) as fin:
		for line in fin:
			words.append (line.strip())

	for word in words[args.from_index:args.till_index]:
		embeddings = list ()
		year_labels = list ()
		paper_ids = list ()
		positions = list ()
		for year in range (args.from_year, args.till_year+1):
			filename = os.path.join (args.word_embeddings_dir, word, f"{year}.tsv")
			if os.path.isfile (filename):
				ids, pos, embeds = read_stats_from_file (filename)
				year_labels.extend ([year] * len (embeds))
				embeddings.extend (embeds)
				paper_ids.extend (ids)
				positions.extend (pos)

		y = np.array (year_labels)
		X = np.array (embeddings)
		X = standardize (X)
		y = (y >= groundtruth[word])
		clf = LogisticRegressionCV(Cs=1, fit_intercept=False, cv=4, random_state=1).fit(X, y)
		probs = clf.predict_proba(X)
		predictions = clf.predict (X)
		frame = make_per_instance_frame(probs, y, predictions, paper_ids, positions)
		frame.to_csv (os.path.join (args.word_embeddings_dir, word,f"{word}.classification.tsv"), sep='\t', header=True, index=False)
		meta_json = make_meta_JSON (word, groundtruth[word], y, predictions)
		with open (os.path.join (args.word_embeddings_dir, word, f"{word}.classification_meta.json"), "w") as fout:
			fout.write (f'{json.dumps (meta_json)}\n')

	return
		

	SEMICOL=';'
	df = pd.read_csv (os.path.join (args.embeddings_dir, args.innovs_file), sep=SEMICOL)
	words = df.word.values.tolist()
	years = df.year.values.tolist()
	i = words.index (args.word)
	
	embeddings_file = os.path.join (args.embeddings_dir, args.words_dir, args.word, 'embeddings.tsv')
	if not os.path.exists (embeddings_file):
		print (f'{embeddings_file} not found')
		return

	y, X = readEmbeddings (embeddings_file)
	X = standardize (X)
	y = (y > years[i])
	clf = LogisticRegressionCV(Cs=1, fit_intercept=False, cv=4, random_state=1).fit(X, y)
	probs = clf.predict_proba(X)
	predictions = clf.predict (X)
	frame = makePerInstanceFrame(probs, y, predictions)
	frame.to_csv (os.path.join (args.embeddings_dir, args.words_dir, args.word, 'classification.csv'), sep=',', header=True, index=False)
	meta_json = makeMetaJSON (args.word, years[i], y, predictions)
	with open (os.path.join (args.embeddings_dir, args.words_dir, args.word, 'classification_meta.json'), 'w') as fout:
		fout.write (f'{json.dumps (meta_json)}\n')

if __name__ == "__main__":
	main (readArgs ())
