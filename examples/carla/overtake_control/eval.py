import time
import argparse
import pandas as pd
import numpy as np
import os
import sys
from monitor.dt_learner import *
import csv
from config import *

FP_LOOKAHEAD = INPUT_WINDOW + 25

def fp_condition(df, start, end):
    """
    Arbitrary condition for safety evaluated on a given interval
    Must be evaluated quickly for overall runtime to be low
    """
    end = min(len(df.index), end)
    for j in range(start, end):
        safe = df.loc[j, "safe"]
        # print(f"Safe at index {j}?: {safe}")
        if not safe:
            return False
    return True

def calc_false_positive(csv_file_path):
	"""
	Count #FP in the csv log.
	A time step is FP if DM chooses SC but it is safe in the next FP_LOOKAHEAD time steps.
	"""
	df = pd.read_csv(csv_file_path)

	num_fp = 0
	total_sc = 0

	for i in range(len(df)):
		row = df.iloc[i]
		dm_output = row["DM_output"]
		# is_safe = row["safe"]

		if dm_output == 0: # if using SC
			total_sc += 1

			# calc flase positive
			current_label = fp_condition(df, i, i + FP_LOOKAHEAD)
			#current_label = condition(df, i + input_window + horizon, i + input_window + horizon + decision_window)
			if current_label:
				num_fp += 1

	return total_sc, num_fp


def scan_normal_csv(csv_file_path):
	"""
	Count #violations without SC
	"""
	total_safe = 0
	total_unsafe = 0
	num_fn = 0

	df = pd.read_csv(csv_file_path)

	# print(df.describe())
	# for y in df.columns:
	# 	print(y, df[y].dtype)

	for i in range(len(df)):
		row = df.iloc[i]
		if row["safe"]:
			total_safe += 1
		else:
			total_unsafe += 1

		if row["safe"] == False and row["controller"] == 1:
			num_fn += 1

	return total_safe, total_unsafe, num_fn

def scan_normal_csv_directory(csv_dir_path):
	total_safe = 0
	total_unsafe = 0
	num_fn = 0
	for root, dirs, files in os.walk(csv_dir_path, topdown=False):
		for name in files:
			csv_file_path = os.path.join(root, name)
			print("  Processing ", csv_file_path)
			t0,t1,t2 = scan_normal_csv(csv_file_path)
			total_safe += t0
			total_unsafe += t1
			num_fn += t2
	return total_safe, total_unsafe, num_fn

def scan_allac_csv_directory(csv_dir_path):
	total_sc = 0
	num_fp = 0

	for root, dirs, files in os.walk(csv_dir_path, topdown=False):
		for name in files:
			csv_file_path = os.path.join(root, name)
			try:
				print("  Processing ", csv_file_path)
				t0,t1 = calc_false_positive(csv_file_path)
				total_sc += t0
				num_fp += t1
			except Exception as e:
				print("  Error Processing ", csv_file_path)
				print(str(e))

	return total_sc, num_fp





if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Evaluate Log File.')
	parser.add_argument("csv_file_dir", help="Path to csv file dir")
	parser.add_argument("-n", "--fn", help="Calculate False Negative",
                    action="store_true")
	parser.add_argument("-p", "--fp", help="Estimate False Positive",
                    action="store_true")
	args = parser.parse_args()

	if not args.fn and not args.fp:
		print("Warning: Need to select -n or -p")

	if args.fn:
		total_safe, total_unsafe, num_fn = scan_normal_csv_directory(args.csv_file_dir)
		print(total_safe, total_unsafe, num_fn)
		print("FN Rate: ", num_fn/total_unsafe if total_unsafe != 0 else 0)
	if args.fp:
		total_sc, num_fp = scan_allac_csv_directory(args.csv_file_dir)
		# total_sc, num_fp = calc_false_positive(args.csv_file_dir)
		print(total_sc, num_fp)
		print("FP Rate: ", num_fp/total_sc if total_sc != 0 else 0)




	