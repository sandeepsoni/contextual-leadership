import argparse
import os
import glob
import json
import pandas as pd
import numpy as np
import scipy
from tqdm import tqdm
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import statsmodels.api as sm
import statsmodels.formula.api as smf


def readArgs ():
	parser = argparse.ArgumentParser (description="Test of time regression")
	parser.add_argument ("--citation-file", type=str, required=True, help="File contains paper ids, publication year and the citation distribution")
	parser.add_argument ("--coefficients-dir", type=str, required=True, help="Directory that contains linguistic coefficients")
	parser.add_argument ("--output-file", type=str, required=True, help="File contains the coefficients for the regression")
	parser.add_argument ("--k", type=int, required=False, default=3, help="offset in years")
	parser.add_argument ("--n", type=int, required=False, default=10, help="test of time offset in years")
	args = parser.parse_args ()
	return args

def read_citations_as_df (filename, k=3, n=10):
	rows = list ()
	with open (filename) as fin:
		for line in fin:
			js = json.loads (line)
			paper_id = js["paper_id"]
			year = int (js["publication_year"])
			k_cites = sum ([js["citation_distribution"].get (str(y), 0) for y in range (year+1, year+k+1)])
			n_cites = sum ([js["citation_distribution"].get (str(y), 0) for y in range (year+1, year+n+1)])
			rows.append ([paper_id, year, np.log (k_cites+1), np.log (n_cites+1)])

	df = pd.DataFrame (rows, columns=["paper_id", "publication_year", f"logk_cites(k={k})", f"logn_cites(n={n})"])
	return df

def read_ling_coefficients_as_df (dirname, k=3, prefix="001"):
	coefficients = list ()
	files = glob.glob (os.path.join (dirname, "*.tsv"))
	for filename in tqdm (files):
		basename = os.path.basename (filename)
		if basename.startswith (prefix):
			ling_df = pd.read_csv (filename, sep="\t")
			title = basename.split (".")[0]
			year = int (title[-4:])
			search_year = year - k
			ling_df = ling_df[ling_df.year == search_year]
			for _, row in ling_df.iterrows():
				coefficients.append ([str(int (row["paper_id"].item())), row["influence"].item()])

	return pd.DataFrame (coefficients, columns=["paper_id", "influence"])

def calculate_per_year_zscore (frame, col_name):
	years = frame["publication_year"].unique()
	list_of_dfs = list ()
	for year in years:
		new_df = frame[frame["publication_year"] == year]
		m = new_df[col_name].mean()
		s = new_df[col_name].std()
		new_df["z_score"] = (new_df[col_name]-m)/s
		list_of_dfs.append(new_df)

	z_score_df = pd.concat (list_of_dfs)
	return z_score_df	

def calculate_quantiles_df (frame, col_name, q=4, q_labels=["q1", "q2", "q3", "q4"], quantiles_col="cites_quantiles"):
	years = frame["publication_year"].unique()
	list_of_dfs = list ()
	for year in years:
		new_df = frame[frame["publication_year"] == year]
		#new_df[quantiles_col] = pd.cut (new_df[col_name], bins=4, labels=q_labels, duplicates="drop")
		new_df[quantiles_col] = pd.qcut (new_df[col_name], q=q, labels=q_labels, duplicates="drop")
		list_of_dfs.append(new_df)

	df = pd.concat (list_of_dfs)
	return df	

def get_one_hot_encoding (series):
	le = LabelEncoder ()
	S = le.fit_transform (series)
	ohe = OneHotEncoder ()
	return ohe.fit_transform (S.reshape (-1, 1)).toarray ()

def main (args):
	cites_df = read_citations_as_df (args.citation_file, k=args.k, n=args.n)
	coeffs_df = read_ling_coefficients_as_df (args.coefficients_dir, k=args.k) 
	overall_df = pd.merge (cites_df, coeffs_df, on="paper_id")
	overall_df = calculate_per_year_zscore (overall_df, col_name = f"logn_cites(n={args.n})")
	overall_df = calculate_quantiles_df (overall_df, col_name=f"logk_cites(k={args.k})", q=[0.0, 0.5, 0.75, 0.9, 1.0], q_labels=["c1", "c2", "c3", "c4"], quantiles_col="cites_quantiles")
	overall_df = calculate_quantiles_df (overall_df, col_name="influence", q=[0.0, 0.5, 0.75, 0.9, 1.0], q_labels=["i1", "i2", "i3", "i4"], quantiles_col="influence_quantiles")

	y = overall_df[[f"z_score"]]
	print ("Intercept only model")	
	X = np.ones ((len(overall_df),1))
	reg = sm.OLS (endog=y, exog=X).fit ()
	print (reg.summary())
	print ("Base model")
	reg = smf.ols (formula="z_score~C(cites_quantiles)", data=overall_df).fit()
	print (reg.summary())	
	ll_reduced = reg.llf
	print ("Experimental model")
	reg = smf.ols (formula="z_score~C(cites_quantiles)+C(influence_quantiles)", data=overall_df).fit()
	print (reg.summary())
	ll_full = reg.llf


	print ("Likelihood ratio test")
	lr_stat = -2 * (ll_reduced - ll_full)
	p_val = scipy.stats.chi2.sf(lr_stat, 3)
	print (f"P-value: {p_val}")

	overall_df.rename (columns={f"logk_cites(k={args.k})":"logk_cites"}, inplace=True)
	reg = smf.ols (formula="z_score~logk_cites", data=overall_df).fit ()
	print (reg.summary())

	reg = smf.ols (formula="z_score~logk_cites+influence", data=overall_df).fit ()
	print (reg.summary())
	
	"""

	#y = overall_df[[f"logn_cites(n={args.n})"]]
	y = overall_df[[f"z-score"]]
	reg_penalty = 1e-16
	print ("Intercept only model")	
	X = np.ones ((len(overall_df),1))
	reg = Ridge (alpha=1e-12, fit_intercept=False).fit (X,y)
	print (f"MSE:{mean_squared_error(y, reg.predict (X)):.4f}")
	print ("Base model")
	
	X = get_one_hot_encoding (overall_df[f"logk_cites(k={args.k})_quantiles"])
	#X = overall_df[[f"logk_cites(k={args.k})"]]
	reg = Ridge (alpha=reg_penalty, fit_intercept=True).fit (X,y)
	print (f"MSE:{mean_squared_error(y, reg.predict (X)):.4f}")
	print ("Experimental model")
	#X = overall_df[[f"logk_cites(k={args.k})", "influence"]]	
	X1 = get_one_hot_encoding (overall_df[f"logk_cites(k={args.k})_quantiles"])
	X2 = get_one_hot_encoding (overall_df[f"influence_quantiles"])
	X = np.hstack ((X1, X2))
	reg = Ridge (alpha=reg_penalty, fit_intercept=True).fit (X,y)
	print (f"MSE:{mean_squared_error(y, reg.predict (X)):.4f}")
	print (reg.coef_)
	"""

if __name__ == "__main__":
	main (readArgs ())
