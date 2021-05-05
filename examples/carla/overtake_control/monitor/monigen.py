import time
import argparse
import pandas as pd
import numpy as np
import os
import sys
from monitor.dt_learner import *
import csv
from config import *
def log_to_csv(log_fname):
    """
    Convert a single .log file (feature vector on each line, space-delimited) to .csv (comma-delimited).
    Requires error table CSV having been generated.
    """
    csv_fname = log_fname.replace('.log', '.csv')
    table_fname = log_fname.replace('.log', '_error_table.csv')
    os.system(f"rm {csv_fname}")
    # Get field names from logfile as every other string is a field name
    with open(log_fname, 'r') as lf:
        first_line = lf.readline()
        first_line = first_line.strip().split()
        fieldnames = []
        for i in range(len(first_line)):
            if i % 2 == 0:
                fieldnames.append(first_line[i])
    # Add a column for labeling
    fieldnames.append('safe')
    # Create CSV with log information and extra column from error table
    with open(log_fname, 'r') as lf, open(table_fname, 'r') as tf:
        with open(csv_fname, 'w') as cf:
            writer = csv.DictWriter(cf, fieldnames=fieldnames)
            writer.writeheader()
            lines = lf.readlines()
            verdict_lines = tf.readlines()
            for i in range(len(lines)):
                line = lines[i].strip().split()
                even = line[::2]
                odd = line[1::2]
                line_dict = dict(zip(iter(even), iter(odd)))

                verdict_line = verdict_lines[i].strip().split(',')
                line_dict['safe'] = verdict_line[-1]
                writer.writerow(line_dict)
    return csv_fname
def write_error_table(f_index, output):
    """
    Generate error table from np array with 0's and 1's: assign each timestep True/False in a CSV for the run.
    """
    with open(f'{SIM_DIR}/{f_index}_error_table.csv', 'a')  as f:
        for j in range(N_SIM_STEP):
            if output[j] == 0: tmp = False
            else: tmp = True
            f.write(f'{j},{tmp}\n')
def read_target_speeds():
    """
    Read init.log and return a python list of lists containing [run index, speed]
    """
    init_log = list()
    with open(f'{SIM_DIR}/init.log', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            init_log.append( [ int(line[0]), float(line[1]) ] )
    return init_log
def process_error_tables():
    """
    Read in falsifier output and write out error table CSV containing a matrix of 1 or 0 (violation) for
    each timestep, for each run listed in init.log.
    """
    init_log = read_target_speeds()
    table_df = pd.read_csv(f'{SIM_DIR}/falsifier.csv')
    table_df = table_df[['point.init_conditions.ego_target_speed[0]', 'rho_0']]
    os.system(f"rm {SIM_DIR}/*_error_table.csv")
    for sample in init_log:
        output = np.ones((N_SIM_STEP, 1))
        i = sample[0] # index
        speed = sample[1] # target speed
        for index, row in table_df.iterrows():
            if abs(float(row['point.init_conditions.ego_target_speed[0]'])-speed) > 1e-13: continue
            time = int(row['rho_0']) % N_SIM_STEP
            output[time] = 0
        # Use index and error matrix to write out a CSV
        write_error_table(i, output)
def create_training_data(csv_file_path, input_window, horizon, decision_window, columns, training_columns, condition):
    """
    Return a dataframe containing tranining data based on the sliding window model
    """
    data = pd.read_csv(csv_file_path)

    # Prepare training dataframe
    training_data_columns = []
    for i in range(input_window):
            for c in training_columns:
                    training_data_columns.append(f"{c}@{i}")
    training_data_columns.append("flag")
    training_data = pd.DataFrame(columns=training_data_columns)
    # Iterate over data: for each sliding window of size input_window look horzion many steps into the future and check condition over decision window.
    # label sequence of data input window according to condition
    t0 = time.time()
    training_data_list = []
    for i in range(len(data) - (decision_window + horizon + input_window -1)):
        if i % 100 == 0: print (f"Creating training data, sliding window at timestep {i}")
        if i % 200 == 0 and i != 0: print (f"Elapsed time: {time.time() - t0} seconds")
        t1 = time.time()
        # collect data within an input_window
        # Must be a python list to handle True/False values. df indexing is inclusive on both ends
        entry = data.loc[i:i+input_window-1, training_columns].to_numpy().flatten().tolist()
        # check condition for the collected data
        t2 = time.time()
        current_label = condition(data, i + input_window + horizon, i + input_window + horizon + decision_window)
        # create new training_data entry
        t3 = time.time()
        entry.append(current_label)
        t4 = time.time()
        training_data_list.append(entry)
        t5 = time.time()
    # Create dataframe from list of lists
    training_data = pd.DataFrame(data=training_data_list, columns=training_data_columns)
    t6 = time.time()
    print(f"Creating training data took {t5 - t0} seconds")
    #print(f"Concatenating last input window took {t2 - t1} seconds")
    #print(f"Concatenating last decision window and checking condition took {t3 - t2} seconds")
    #print(f"Creating last training data entry list took {t4 - t3} seconds")
    #print(f"Creating last training data entry dataframe and appending took {t5 - t4} seconds")
    #print(f"Creating new dataframe from list took {t6 - t5} seconds")
    # Postprocess data
    print(f"Before postprocessing: {training_data.flag.value_counts()}")
    training_data = postprocess_data(training_data)
    print(f"After postprocessing: {training_data.flag.value_counts()}")
    t7 = time.time()
    print(f"Postprocessing data took {t7 - t6} seconds")
    # print(training_data)
    return training_data

def postprocess_data(data):
    """
    Take in a training data dataframe and process it
    Label all vectors within a certain euclidean distance of a False vector as also False.
    See config.py for MIN_FEAT_DIST definition
    """

    # Construct a list of functions to map across the dataframe, one for each False (violation) row
    data_no_flags = data.drop("flag", axis=1)  # Default inplace=False
    false_rows = data_no_flags[data["flag"] == False] # All rows with flag False (new df)
    false_rows = false_rows.to_numpy()
    relabeling_fns = []  # List of functions that take in a row and return True/False
    for fr in false_rows:
        """
        def fn(row):
            n = np.linalg.norm(row.drop("flag").to_numpy() - fr)
            print(n)
            return not (n < MIN_FEAT_DIST)
        """
        # Speed up by keeping the numpy-converted array somewhere else and getting it by index?
        # Can use apply(raw=True to pass array but must remove flag col first)
        fn = lambda row: not (np.linalg.norm(row - fr) < MIN_FEAT_DIST)
        relabeling_fns.append(fn)

    # Apply each function row-wise to create a new column with labels indicating whether to revise flag to False
    i = 0
    relabeling_cols = [] # List of tuples containing (name, column data)
    for fn in relabeling_fns:
        if i % 20 == 0: print(f"Mapping relabeling function {i} out of {len(relabeling_fns)}")
        relabeling_cols.append(("rel" + str(i), data_no_flags.apply(fn, axis=1, raw=True)))
        i += 1
    # Append the relabeling columns
    for n, c in relabeling_cols:
        data[n] = c

    # Look at all new label columns and revise flag column if any relabeling entry or initial flag is False
    relabeling_col_names = [n for n, c in relabeling_cols] + ["flag"]
    data["flag"] = ~((~data[relabeling_col_names]).any(axis=1))
    # Remove relabeling columns
    relabeling_col_names.remove("flag")
    data.drop(relabeling_col_names, axis=1, inplace=True)
    return data

def create_monitor_wrapper(dt_import_path, feature_names, model_prefix):
    # sanity checks
    assert model_prefix in ("ego", "other"), "Monitor needs model for 'ego' and 'other' car."
    if not any("other" in e for e in feature_names):
        print("Warning: Did not find any feature names containing 'other' for ego monitor. Is it intended?")


    print(f"Create monitor file at {dt_import_path}/monitor_{model_prefix}.py")
    code_file = open(f"{dt_import_path}/monitor_{model_prefix}.py", "w")

    indent = "    "
    # imports
    #code_file.write("import importlib\n")
    code_file.write("import os\n")
    code_file.write("import numpy as np\n")
    code_file.write("from joblib import load\n")
    # code_file.write(f"from examples.carla.overtake_control.simpath.dt import dt\n\n")

    # global variables
    code_file.write("window_data = [] \n")
    # code_file.write("dt_map = {}\n\n")

    # function def
    code_file.write("def check(input_map, input_window, reload_dt):\n\n")

    # global variables definitions
    code_file.write(indent + "dt_map = {}\n")
    code_file.write(indent + "global window_data\n\n")

    # filled buffer
    code_file.write(indent + "# check if data has reached window size\n")
    code_file.write(indent + "window_fill_size = len(window_data)\n")
    code_file.write(indent + "if window_fill_size < input_window:\n")
    code_file.write(indent + indent + "# expand window_data with input_map\n")
    code_file.write(indent + indent + "window_data.append(input_map)\n")
    code_file.write(indent + indent + "return 1\n\n")

    # Implement FIFO
    code_file.write(indent + "# FIFO behavior: Buffer of size input_window\n")
    code_file.write(indent + "window_data.pop(0)\n")
    code_file.write(indent + "window_data.append(input_map)\n\n")

    # dt map initialization
    code_file.write(indent + "# initialize map if not initialized\n")
    code_file.write(indent + "if not dt_map:\n")
    code_file.write(indent + indent + "for i in range(window_fill_size):\n")
    code_file.write(indent + indent + indent + "for key in window_data[i].keys():\n")
    code_file.write(indent + indent + indent + indent + "name = f\"{key}@{i}\"\n")
    code_file.write(indent + indent + indent + indent + "dt_map[name] = window_data[i][key]\n\n")
    code_file.write(indent + "#print(dt_map)\n\n")

    code_file.write(indent + "# need to convert to X format\n")
    code_file.write(indent + "X = []\n")
    code_file.write(indent + "features = " + str(feature_names) + "\n")
    code_file.write(indent + "for feature in features:\n")
    code_file.write(indent + indent + "X.append(dt_map[feature])\n")
    code_file.write(indent + "X = np.array(X)\n")
    code_file.write(indent + "X = np.expand_dims(X, axis=0)\n")

    code_file.write(indent + "dts = []\n")
    code_file.write(indent + f"if not os.path.isdir('./examples/carla/overtake_control/monitor/dt_{model_prefix}'): return 1\n")
    code_file.write(indent + f'prev_tree_files = os.listdir("./examples/carla/overtake_control/monitor/dt_{model_prefix}")\n')
    code_file.write(indent + "for fname in prev_tree_files:  # tree_0.joblib\n")
    code_file.write(indent + indent + 'if fname.endswith("joblib"):\n')
    code_file.write(indent + indent + indent + f'dts.append(load(f"./examples/carla/overtake_control/monitor/dt_{model_prefix}/{{fname}}"))\n')

    code_file.write(indent + "# if no tree is found, use AC\n")
    code_file.write(indent + "if not dts: return 1\n")
    code_file.write("\n")

    code_file.write(indent + "for dt in dts:\n")
    code_file.write(indent + indent + "verdict = dt.predict(X)[0]\n")
    code_file.write(indent + indent + "if verdict == 0: return 0\n")
    code_file.write(indent + "return 1\n")
    #code_file.write(indent + "a = 0.1\n")
    #code_file.write(indent + "v_sum = 0\n")
    #code_file.write(indent + "for i in range(len(dts)):\n")
    #code_file.write(indent + indent + "verdict = dts[i].predict(X)[0]\n")
    #code_file.write(indent + indent + "v_sum += ( a**(float(i!=0)) )* ( (1-a)**(len(dts)-1-i) ) * float(verdict)\n")

    #code_file.write(indent + "if v_sum >= 0.5:\n")
    #code_file.write(indent + indent + "return 1\n")
    #code_file.write(indent + "else:\n")
    #code_file.write(indent + indent + "return 0\n")

    code_file.close()


def process_log_files():
    """
    Convert all 0.log, 1.log, etc. files to CSV files.
    """
    simulation_files = os.listdir(f"{data_dir}")
    for f in simulation_files:
        if f.endswith(".log") and not f.startswith('init'):
            file_path = f"{data_dir}/{f}"
            file_path = log_to_csv(file_path)
def move_csv_files():
    max_iter = -1
    simulation_files = os.listdir(f"{data_dir}")
    for f in simulation_files:
        if f.endswith("_run"):
            iteration = f.split("_")[0]
            max_iter = max([max_iter, int( iteration )])
    folder_name = f'{max_iter+1}_run'
    os.system(f'mkdir {data_dir}/{folder_name}')
    os.system(f'mv {data_dir}/*.csv {data_dir}/{folder_name}')

def remove_files():
    simulation_files = os.listdir(f"{data_dir}")
    for f in simulation_files:
        if f.endswith("_error_table.csv") or f.endswith(".log") or f.endswith("falsifier.csv"):
            file_path = f"{data_dir}/{f}"
            os.system(f'rm {file_path}')
    move_csv_files()
def get_max_run_iter():
    """
    Get the highest index of folders titled 0_run, 1_run, etc.
    """
    simulation_files = os.listdir(f"{data_dir}")
    max_iter = -1
    for f in simulation_files:
        if f.endswith("_run"):
            iteration = f.split("_")[0]
            max_iter = max([max_iter, int( iteration )])
    return max_iter
def get_training_data_max_index():
    training_data_files = os.listdir(f"{data_dir}/training_data")
    max_iter = -1
    for f in training_data_files:
        if f.startswith("training_data_"):
            iteration = f.split("_")[-1]
            iteration = iteration.split(".csv")[0]
            max_iter = max([max_iter, int( iteration )])
    return max_iter

def generate(data_dir, column_names, training_column_names, condition, input_window=2, horizon=2, decision_window=2):
    # Iterate over all simulation files
    if os.path.exists(f'{data_dir}/falsifier.csv'):
        print(f"Generating error tables from falsifier.csv...")
        process_error_tables()
        print(f"Generating labeled CSVs from log files...")
        process_log_files()
        print(f"Cleaning up temporary files...")
        remove_files()
    training_data_list = []
    f = str( get_max_run_iter() ) + "_run"
    sub_files = os.listdir(f"{data_dir}/{f}")
    # For each iteration in the most recent run, generate training data from the labeled CSV
    for sf in sub_files:
        if sf.endswith(".csv"):
            print(f"Creating training data from {f}/{sf} ...")
            file_path = f"{data_dir}/{f}/{sf}"
            training_data_list.append(create_training_data(file_path, input_window, horizon, decision_window, column_names, training_column_names, condition))
    start_index = get_training_data_max_index() + 1
    feature_names = []
    class_names = ['flag']
    # Write out training data dataframes as CSV
    for i in range(len(training_data_list)):
        training_data_list[i].to_csv(f"{data_dir}/training_data/training_data_{i+start_index}.csv",index=False,header=False)
        if i ==0:
            feature_names = list(training_data_list[i].columns)[:-1]
    os.system(f"rm {data_dir}/training_data/training_data.csv")
    os.system(f"cat {data_dir}/training_data/*csv > {data_dir}/training_data/training_data.csv")

    learn_dt(f"{data_dir}/training_data/training_data.csv", class_names, feature_names, "dt",False, data_dir)

    create_monitor_wrapper(data_dir)

def generate_from_scratch(data_dir, column_names, training_column_names, condition, input_window=2, horizon=2, decision_window=2, model_prefix="ego"):
        # Iterate over all simulation files
        if os.path.exists(f'{data_dir}/falsifier.csv'):
            print(f"Generating error tables from falsifier.csv...")
            process_error_tables()
            print(f"Generating labeled CSVs from log files...")
            process_log_files()
            print(f"Cleaning up temporary files...")
            remove_files()
        simulation_files = os.listdir(f"{data_dir}")
        training_data_list = []
        f = str( get_max_run_iter() ) + "_run"
        sub_files = os.listdir(f"{data_dir}/{f}")
        for sf in sub_files:
            if sf.endswith(".csv"):
                print(f"Creating training data from {f}/{sf} ...")
                file_path = f"{data_dir}/{f}/{sf}"
                training_data_list.append(create_training_data(file_path, input_window, horizon, decision_window, column_names, training_column_names, condition))

        os.system(f"rm -r {data_dir}/training_data")
        os.system(f"mkdir {data_dir}/training_data")
        feature_names = []
        class_names = ['flag']
        for i in range(len(training_data_list)):
                if i ==0:
                        training_data_list[i].to_csv(f"{data_dir}/training_data/training_data_{i}.csv",index=False)
                        feature_names = list(training_data_list[i].columns)[:-1]
                else:
                        training_data_list[i].to_csv(f"{data_dir}/training_data/training_data_{i}.csv",index=False,header=False)
        os.system(f"cat {data_dir}/training_data/*csv > {data_dir}/training_data/training_data.csv")


        # generate dt for ego or other
        print(f"Generating DT: {model_prefix} -> ")

        learn_dt(f"{data_dir}/training_data/training_data.csv", class_names, feature_names, True ,False, OUTPUT_MONITOR_DIR, model_prefix)

        create_monitor_wrapper(OUTPUT_MONITOR_DIR, feature_names, model_prefix)


# DEBUGGING

columns = ['v', 'other_heading', 'other_distance', 'waypoint_25_dtc', 'waypoint_20_dtc', 'waypoint_15_dtc', 'waypoint_10_dtc', 'waypoint_5_dtc', 'waypoint_0_dtc', 'safe']
training_columns = ['v', 'other_heading', 'other_distance', 'waypoint_25_dtc', 'waypoint_20_dtc', 'waypoint_15_dtc', 'waypoint_10_dtc', 'waypoint_5_dtc', 'waypoint_0_dtc']
data_dir = SIM_DIR

# columns = ['v', 'waypoint_5_dtc', 'waypoint_4_dtc', 'waypoint_3_dtc', 'waypoint_2_dtc', 'waypoint_1_dtc', 'waypoint_0_dtc', 'safe']
# training_columns = ['v', 'waypoint_5_dtc', 'waypoint_4_dtc', 'waypoint_3_dtc', 'waypoint_2_dtc', 'waypoint_1_dtc', 'waypoint_0_dtc']
# data_dir = SIM_DIR

def condition(df, start, end):
    """
    Arbitrary condition for safety evaluated on a given interval
    Must be evaluated quickly for overall runtime to be low
    """
    for j in range(start, end):
        safe = df.loc[j, "safe"]
        # print(f"Safe at index {j}?: {safe}")
        if not safe:
            return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Monitor File.')
    parser.add_argument("prefix", help="The agent the monitor controls. Must be ego or other.")
    args = parser.parse_args()


    # Generate DT from latest run
    generate_from_scratch(data_dir,columns,training_columns, condition,INPUT_WINDOW,5,20, model_prefix=args.prefix)
    #generate(data_dir,columns,training_columns, condition,INPUT_WINDOW,5,20)
