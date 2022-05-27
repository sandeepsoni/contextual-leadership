import argparse
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

def getArgs ():
	parser = argparse.ArgumentParser (description="Calculate the spearman correlation between citations and linguistic influence")
	parser.add_argument ("--input-file", type=str, required=True, help="File contains columnwise data for citations and linguistic influence")
	args = parser.parse_args ()
	return args

def main (args):
	quantiles = 4
	df = pd.read_csv (f"{args.input_file}", sep="\t")
	df["log_cites"] = np.log (df["n_cites"] + 1)
	print (f"Spearman Correlation: {spearmanr (df['log_cites'], df['ling_coeff'])}")
	df["ling_coeff_quantiles"] = pd.qcut(df.ling_coeff, quantiles, labels=[f"q{i+1}" for i in range (quantiles)], duplicates="drop")
	for quantile in sorted(df["ling_coeff_quantiles"].unique()):
		print (f"Linguistic coefficient quantile: {quantile}, Avg. log citations={df[df['ling_coeff_quantiles'] == quantile]['log_cites'].mean()}, Median log citations={df[df['ling_coeff_quantiles'] == quantile]['log_cites'].median()}")

	min_cites=10 #@param {type:"number"}
	print ("Quantile | Citations(Mean) | Citations(Std) | Citations (Median) |")
	for quantile in ["q1", "q2", "q3", "q4"]:
		print (quantile, df[(df["n_cites"] >= min_cites) & (df["ling_coeff_quantiles"] == quantile)]["log_cites"].mean(), df[(df["n_cites"] >= min_cites) & (df["ling_coeff_quantiles"] == quantile)]["log_cites"].std(), df[(df["n_cites"] >= min_cites) & (df["ling_coeff_quantiles"] == quantile)]["log_cites"].median())
	
	print ("Exponentially spaced bins")
	df["ling_coeff_quantiles"] = pd.qcut(df.ling_coeff, q=[0, .5, 0.9, 0.99, 1.0], labels=["q1", "q2", "q3", "q4"], duplicates="drop")
	for quantile in sorted(df["ling_coeff_quantiles"].unique()):
		print (f"Linguistic coefficient quantile: {quantile}, Avg. log citations={df[df['ling_coeff_quantiles'] == quantile]['log_cites'].mean()}, Median log citations={df[df['ling_coeff_quantiles'] == quantile]['log_cites'].median()}")
	
	min_cites=20 #@param {type:"number"}
	print ("Quantile | Citations(Mean) | Citations(Std) | Citations (Median) |")
	for quantile in ["q1", "q2", "q3", "q4"]:
		print (quantile, df[(df["n_cites"] >= min_cites) & (df["ling_coeff_quantiles"] == quantile)]["log_cites"].mean(), df[(df["n_cites"] >= min_cites) & (df["ling_coeff_quantiles"] == quantile)]["log_cites"].std(), df[(df["n_cites"] >= min_cites) & (df["ling_coeff_quantiles"] == quantile)]["log_cites"].median())
	

if __name__ == "__main__":
	main (getArgs ())
