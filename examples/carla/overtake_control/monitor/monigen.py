import time
import pandas as pd
import numpy as np
import os
import sys
from monitor.dt_learner import *
import csv
from config import *
def log_to_csv(log_fname):
    csv_fname = log_fname.replace('.log', '.csv')
    table_fname = log_fname.replace('.log', '_error_table.csv')
    os.system(f"rm {csv_fname}")
    with open(log_fname, 'r') as lf:
        first_line = lf.readline()
        first_line = first_line.strip().split()
        fieldnames = []
        for i in range(len(first_line)):
            if i % 2 == 0:
                fieldnames.append(first_line[i])
    fieldnames.append('safe')
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
    with open(f'{SIM_DIR}/{f_index}_error_table.csv', 'a')  as f:
        for j in range(N_SIM_STEP):
            if output[j] == 0: tmp = False
            else: tmp = True
            f.write(f'{j},{tmp}\n')

def process_error_tables():
    table_df = pd.read_csv(f'{SIM_DIR}/falsifier.csv')
    table_df = table_df[['rho', 'rho_0']]
    os.system(f"rm {SIM_DIR}/*_error_table.csv")
    last_time = -1
    output = np.ones((N_SIM_STEP, 1))
    i = 0
    for index, row in table_df.iterrows():
        time = int(row['rho_0']) % N_SIM_STEP
        output[time] = 0
        if time < last_time:
            i = int(row['rho_0']) // N_SIM_STEP
            write_error_table(i-1, output)
            output = np.ones((N_SIM_STEP, 1))
            output[time] = 0
        last_time = time
    write_error_table(i, output)
def create_training_data(csv_file_path, input_window, horizon, decision_window, columns, training_columns, condition):
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
        for i in range(len(data) - (decision_window + horizon + input_window -1)):
                if i % 20 == 0: print (i) 
                # collect data within an input_window
                current_input_window_data = pd.DataFrame(columns= columns)
                for j in range(i,i + input_window):
                        current_input_window_data.loc[j] = data.loc[j,columns]
                current_input_window_data.reset_index(drop=True)
                # print(current_input_window_data)

                # check condition for the collected data
                current_decision_window_data = pd.DataFrame(columns=columns)
                for j in range(i + input_window + horizon, i + input_window + horizon + decision_window):
                        current_decision_window_data.loc[j] = data.loc[j,columns]
                # print(current_decision_window_data)
                current_label = not condition(current_decision_window_data)

                # create new training_data entry 
                entry = []
                for j in range(i,i+len(current_input_window_data)):
                        for c in training_columns:
                                entry.append(current_input_window_data.loc[j,[c]].values[0])
                
                entry.append(current_label)
                
                # if not (training_data.loc[:,training_data.columns != "flag"] == entry[:-1]).all(1).any():
                panda_entry = pd.DataFrame([entry], columns=training_data_columns)
                training_data = training_data.append(panda_entry,ignore_index=True)
                
                                 
        # print(training_data)
        return training_data


def create_monitor_wrapper(dt_import_path):
        print(dt_import_path)
        code_file = open(f"{dt_import_path}/monitor.py", "w")

        indent = "      "
        # imports
        code_file.write("import importlib\n")
        code_file.write(f"from dt import dt\n\n")
        
        # global variables 
        code_file.write("window_data = [] \n")
        # code_file.write("dt_map = {}\n\n")
        

        # function def
        code_file.write("def check(input_map, input_window, reload_dt):\n\n")

        # global variables definitions
        code_file.write(indent+"dt_map = {}\n")
        code_file.write(indent+"global window_data\n\n")

        # filled buffer
        code_file.write(indent+"# check if data has reached window size\n")
        code_file.write(indent+"window_fill_size = len(window_data)\n")
        code_file.write(indent+"if window_fill_size < input_window:\n")
        code_file.write(indent+indent+"# expand window_data with input_map\n")
        code_file.write(indent+indent+"window_data.append(input_map)\n")
        code_file.write(indent+indent+"return False\n\n")

        # Implement FIFO
        code_file.write(indent+"# FIFO behavior: Buffer of size input_window\n")
        code_file.write(indent+"window_data.pop(0)\n")
        code_file.write(indent+"window_data.append(input_map)\n\n")

        # dt map initialization
        code_file.write(indent+"# initialize map if not initialized\n")
        code_file.write(indent+"if not dt_map:\n")
        code_file.write(indent+indent+"for i in range(window_fill_size):\n")
        code_file.write(indent+indent+indent+"for key in window_data[i].keys():\n")
        code_file.write(indent+indent+indent+indent+"name = f\"{key}@{i}\"\n")
        code_file.write(indent+indent+indent+indent+"dt_map[name] = window_data[i][key]\n\n")
        code_file.write(indent+"print(dt_map)\n\n")

        # reload dt
        code_file.write(indent+"# import decision tree\n")
        code_file.write(indent+"if reload_dt:\n")
        code_file.write(indent+indent+"importlib.reload(dt)\n\n")

        # compute and return verdict
        code_file.write(indent+"verdict = dt.execute(dt_map)\n")
        code_file.write(indent+"return verdict")
        
        code_file.close()

def generate(data_dir, column_names, training_column_names, condition, input_window=2, horizon=2, decision_window=2):
        # Iterate over all simulation files
        process_error_tables()
        simulation_files = os.listdir(f"{data_dir}")
        training_data_list = []
        for f in simulation_files:
                if f.endswith(".log"):
                        print(f"Creating training data from {f} ...")
                        file_path = f"{data_dir}/{f}"
                        file_path = log_to_csv(file_path)
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

        learn_dt(f"{data_dir}/training_data/training_data.csv", class_names, feature_names, "dt",False, data_dir)

        create_monitor_wrapper(data_dir)


# DEBUGGING 
columns = ['time', 'dtc', 'safe']
training_columns = ['time', 'dtc']
data_dir = SIM_DIR 

def condition(df):
    return (df['safe'] == False).any()

generate(data_dir,columns,training_columns, condition,3,3,3)



