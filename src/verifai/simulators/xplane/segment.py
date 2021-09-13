import argparse
import numpy as np
import pandas as pd
import pickle
import os


input_window = 5
decision_window = 2
horizon = 2
sample_rate = 1


prediction_function = lambda i, d, ctes: all([c < 2 for c in ctes[i:i+d]])

def segment_data():
    traces = []
    valid_samples, sample_count = 0, 0
    for file in os.listdir(os.path.join(args.dir, 'traces')):
        filename = os.path.join(args.dir, 'traces', file)
        if filename.endswith('.pkl'):
            file_b = os.path.splitext(filename)[0]
            data = pd.read_pickle(filename)

            ctes = [value for _, value in data['cte'].items()]

            # select indices from data to keep
            indices_to_keep = [0]
            prev_time = data.at[0, 'time']
            for i, time in data['time'].items():
                if i == 0:
                    continue
                if time - prev_time > sample_rate:
                    indices_to_keep.append(i)
                    prev_time = time

            ctes = [ctes[i] for i in indices_to_keep]

            n_items = len(indices_to_keep)
            for i in range(n_items - (decision_window + horizon + input_window)):
                trace = []
                if i + input_window >= n_items - (decision_window + horizon + input_window):
                    break 
                for j in range(i, i + input_window):
                    idx = indices_to_keep[j]
                    for col in data.columns:
                        trace.append((col, data.at[idx, col]))
                index = i + horizon + input_window
                prediction = prediction_function(index, decision_window, ctes)
                traces.append((trace, prediction))
                valid_samples += int(prediction)
                sample_count += 1

    return traces, valid_samples, sample_count



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', help='directory with data', default='./data')
    args = parser.parse_args()

    training_data, valid_samples, n_samples = segment_data()

    with open('training_data.pkl', 'wb') as f:
    	pickle.dump(training_data, f)

    print(valid_samples, n_samples)
