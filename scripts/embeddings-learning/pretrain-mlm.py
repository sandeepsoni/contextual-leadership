import argparse
import math
import argparse
import os
from datasets import load_dataset
from itertools import chain
import transformers
from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling

class LM (object):
	def __init__ (self, model_checkpoint):
		self.model_checkpoint = model_checkpoint
		self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
		self.model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

	def add_special_tokens (self, tokens):
		self.tokenizer.add_tokens (tokens, special_tokens=True)
		self.model.resize_token_embeddings (len(self.tokenizer))	

def tokenize(examples, tokenizer):
	return tokenizer(examples["full_text"])

def group_texts(examples, block_size):
	# Concatenate all texts.
	#concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
	#print (examples.keys())
	concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}

	#concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
	total_length = len(concatenated_examples[list(examples.keys())[0]])
	# We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
	# customize this part to your needs.
	total_length = (total_length // block_size) * block_size
	# Split by chunks of max_len.
	result = {
		k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
		for k, t in concatenated_examples.items()
	}
	result["labels"] = result["input_ids"].copy() # I don't remeber why I needed this line?
	return result
		
def readArgs ():
	parser = argparse.ArgumentParser (description="Finetune a MLM model on domain text")
	parser.add_argument ("--data-file", type=str, required=True, help="File contains text to train the model")
	parser.add_argument ("--model-checkpoint", type=str, required=False, default="bert-base-uncased", help="Download and cache the given model")
	parser.add_argument ("--checkpoints-dir", type=str, required=True, help="The checkpoints are saved here")
	parser.add_argument ("--num-train-epochs", type=int, required=False, default=5, help="The number of epochs to train")
	return parser.parse_args ()

def main (args):
	transformers.logging.set_verbosity_info()
	dataset = load_dataset(
		"json",
		data_files=args.data_file
	)
	lm = LM (args.model_checkpoint)
	tokenized_datasets = dataset.map (
		tokenize,
		batched=True,
		num_proc=4,
		remove_columns=["paper_id"],
		fn_kwargs=dict(tokenizer=lm.tokenizer)
	)

	lm_datasets = tokenized_datasets.map(
		group_texts,
		batched=True,
		batch_size=8,
		num_proc=4,
		fn_kwargs=dict (block_size=lm.tokenizer.model_max_length)
	)

	training_args = TrainingArguments(
		args.checkpoints_dir,
		evaluation_strategy = "epoch",
		num_train_epochs=args.num_train_epochs,
		learning_rate=2e-5,
		weight_decay=0.01,
		save_steps=1000,
		logging_steps=1000,
		save_total_limit=10,
		per_device_train_batch_size=16,
		per_device_eval_batch_size=16,
	)

	data_collator = DataCollatorForLanguageModeling(tokenizer=lm.tokenizer, mlm_probability=0.15)

	trainer = Trainer(
		model=lm.model,
		tokenizer=lm.tokenizer,
		args=training_args,
		train_dataset=lm_datasets["train"],
		#eval_dataset=lm_datasets["validation"],
		data_collator=data_collator,
	)
	
	#train_results = trainer.train(True)
	train_results = trainer.train()
	trainer.save_state ()

if __name__ == "__main__":
	main (readArgs ())
