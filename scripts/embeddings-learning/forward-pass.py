import argparse
import transformers
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer
import torch
import json

class LM (object):
	def __init__ (self, model_checkpoint):
		self.model_checkpoint = model_checkpoint
		self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
		self.model = AutoModelForMaskedLM.from_pretrained(model_checkpoint, output_hidden_states=True)

def split2chunks (encoded_input, split_len=510):
	# Break into smaller chunks
	input_ids_chunks = list(encoded_input['input_ids'][0].split(split_len))
	mask_chunks = list(encoded_input['attention_mask'][0].split(split_len))
    
	for i in range (len (input_ids_chunks)):
		pad_len = 510 - input_ids_chunks[i].shape[0]
		# check if tensor length satisfies required chunk size
		if pad_len > 0:
			# if padding length is more than 0, we must add padding
			input_ids_chunks[i] = torch.cat([
				input_ids_chunks[i], torch.Tensor([0] * pad_len)
			])
			mask_chunks[i] = torch.cat([
				mask_chunks[i], torch.Tensor([0] * pad_len)
			])
		# Append the CLS token (id=101) and the SEP token (id=102)
		input_ids_chunks[i] = torch.cat([
			torch.Tensor([101]), input_ids_chunks[i], torch.Tensor ([102])
		])
            
		# Add attention masks
		mask_chunks[i] = torch.cat([
			torch.Tensor([1]), mask_chunks[i], torch.Tensor([1])
		])
        
	# Now aggregate into one example
	input_ids = torch.stack(input_ids_chunks)
	attention_mask = torch.stack(mask_chunks)
        
	input_dict = {
		'input_ids': input_ids.long().clone().detach(), #torch.tensor(input_ids.long()),
		'attention_mask': attention_mask.int().clone().detach() #torch.tensor(attention_mask.int())
	}
	return input_dict

def get_flattened_embeddings (outputs, attention_mask):
	# Let's concatenate the representation of the final four layers
	embeddings = torch.cat((outputs.hidden_states[-1], #12th hidden layer
							outputs.hidden_states[-2], #11th hidden layer
							outputs.hidden_states[-3], #10th hidden layer
							outputs.hidden_states[-4]), dim=2)
	embeddings = torch.flatten (embeddings[:,1:-1,:], start_dim=0, end_dim=1)
	num_nonzero = (attention_mask[:,1:-1].flatten() == 0).nonzero(as_tuple=True)[0].size()[0]
	if num_nonzero == 0:
		index = None
	else:
		index = (attention_mask[:,1:-1].flatten() == 0).nonzero(as_tuple=True)[0][0].item()
	return embeddings[0:index, :]

def tokens_generator (toks):
	last_token = ""
	i = 0
	token_start = 0
	while i < len (toks):
		if i == 0:
			last_token = toks[i]
			token_start = i
		elif toks[i].startswith ("##"):
			last_token = last_token + toks[i][2:]
		else:
			yield token_start, i, last_token
			last_token = toks[i]
			token_start = i
		i += 1

	if len (last_token) > 0:
		yield token_start, i, last_token

def readArgs ():
	parser = argparse.ArgumentParser (description="Run a forward pass and get the contextual embeddings")
	parser.add_argument ("--model-checkpoint", type=str, required=True, help="Model checkpoint")
	parser.add_argument ("--text-file", type=str, required=True, help="JSONL file contains text")
	parser.add_argument ("--embeddings-file", type=str, required=True, help="Embeddings file as TSV")
	args = parser.parse_args ()
	return args

def main (args):
	#lm = LM ("../checkpoints/contextual-word-embeddings/checkpoint-9000/")
	lm = LM (args.model_checkpoint)
	with open (args.text_file) as fin, open (args.embeddings_file, "w") as fout:
		for line in fin:
			js = json.loads (line.strip())
			# extract text
			text = js["full_text"] # extract additional metadata for later
			if len (text) == 0:
				continue
			paper_id = js["paper_id"]
			# encode the entire text
			encoded_input = lm.tokenizer(text,
										 add_special_tokens=False,
										 return_tensors='pt')
        	
			with torch.no_grad ():
				# print (encoded_input["input_ids"].size()) # contains approx. these many tokens
				input_dict = split2chunks (encoded_input)
				outputs = lm.model(**input_dict)
				embeddings = get_flattened_embeddings (outputs, input_dict["attention_mask"])
				wordpieces = lm.tokenizer.convert_ids_to_tokens(encoded_input["input_ids"][0])
				tokens = [token for token in tokens_generator(wordpieces)]
				tokenized_text = [token for _,_,token in tokens]
				token_boundaries = [(start, ended) for start, ended, _ in tokens]
				token_embeddings = torch.stack([embeddings[start:end,:].mean(dim=0) for start, end in token_boundaries])
        
			for i, token in enumerate (tokenized_text):
				string_rep = ' '.join(list(map(str,token_embeddings[i].tolist())))
				fout.write (f'{paper_id}\t{i}\t{token}\t{string_rep}\n')	

if __name__ == "__main__":
	main (readArgs ())
