import argparse
import json
import pickle
from tqdm import tqdm
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel

def readArgs ():
	parser = argparse.ArgumentParser (description="create and store a bow representation of the entire corpus")
	parser.add_argument ("--input-filename", type=str, required=True, help="File contains the text from the entire document collection")
	parser.add_argument ("--dict-filename", type=str, required=True, help="File contains the dictionary mapping words to ids and vice versa")
	parser.add_argument ("--max-vocab-size", type=int, required=False, default=10000, help="The maximum size of the vocabulary")
	parser.add_argument ("--num-topics", type=int, required=False, default=10, help="The number of topics")
	parser.add_argument ("--output-filename", type=str, required=True, help="File contains the bow representation of all documents in the document collection")
	args = parser.parse_args ()
	return args

def iter_tokens (filename, max_token_size=2):
	with open (filename) as fin:
		for line in tqdm (fin):
			js = json.loads (line)
			paper_id = js["paper_id"]
			tokens = [token.lower() for token in js["tokenized_text"] if token.isalpha() and len (token) > max_token_size]	
			yield paper_id, tokens

def iter_bows (filename, dictionary):
	for paper_id, tokens in iter_tokens (filename):
		yield paper_id, dictionary.doc2bow (tokens)

def make_dictionary (filename, max_batch_size=5000):
	dictionary = Dictionary ()
	batch = list ()
	for _, tokens in iter_tokens (filename):
		if len (batch) < max_batch_size:
			batch.append (tokens)
		else:
			dictionary.add_documents (batch)
			batch = list ()
			batch.append (tokens)

	if len (batch) < max_batch_size:
		dictionary.add_documents (batch)

	return dictionary

def make_lda_model (filename, dictionary, num_topics=10, max_batch_size=5000):
	batch = list ()
	lda = LdaModel (num_topics=num_topics, id2word=dictionary, eta="symmetric")
	for _, bow in iter_bows (filename, dictionary):
		if len (batch) < max_batch_size:
			batch.append (bow)
		else:
			lda.update (batch)
			batch = list ()
			batch.append (bow)

	if len (batch) < max_batch_size:
		lda.update (batch)

	return lda

def main (args):
	dictionary = make_dictionary (args.input_filename)
	dictionary.filter_extremes (no_below=10, no_above=0.5, keep_n=args.max_vocab_size)
	dictionary.compactify ()
	lda = make_lda_model (args.input_filename, dictionary, num_topics=args.num_topics)
	dictionary.save_as_text (args.dict_filename)	
	with open (args.output_filename, "w") as fout:
		for paper_id, bow in iter_bows(args.input_filename, dictionary):
			js = dict ()
			js["paper_id"] = paper_id
			js["bow"] = bow
			js["topics"] = [(topic_id, str(prob)) for topic_id, prob in lda.get_document_topics (bow)]
			fout.write (f"{json.dumps (js)}\n")
	
if __name__ == "__main__":
	main (readArgs ())
