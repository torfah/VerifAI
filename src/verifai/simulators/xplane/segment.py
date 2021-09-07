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
    training_data = []
    for file in os.listdir(os.path.join(args.dir, 'traces')):
        if file.endswith('.pkl'):
            filename = os.path.splitext(file)[0]
            data = pd.read_pickle(file)

            items = []
            for col in data.columns:
                if col != 'cte':
                    items.append([float(value) for _, value in data[col].items()])

            ctes = [value for _, value in data['cte'].items()]

            # select indices from data to keep
            indices_to_keep = [0]
    		prev_time = times[0]
    		for i, time in enumerate(times[1:]):
    			if time - prev_time > sample_rate:
    				indices_to_keep.append(i+1)
    				prev_time = time

            indexed_items = []
            for item in items:
                indexed_items.append([item[i] for i in indices_to_keep])

            assert all([len(items[0]) == len(item) for item in items]

            n_items = len(items[0])
    		for i in range(n_items - (decision_window + horizon + input_window)):
    			concat_data = []
    			for j in range(i, i + input_window):
    				concat_data.append([item[j] for item in items])
    			index = i + horizon + input_window
    			prediction = prediction_function(index, decision_window, ctes)
    			train_data.append((concat_data, prediction))
    			valid_samples += int(prediction)
    			sample_count += 1

    return training_data, valid_samples, sample_count



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', help='directory with data', default='./data')
    args = parser.parse_args()

    training_data, valid_samples, n_samples = segment_data()

    with open('training_data.pkl', 'wb') as f:
    	pickle.dump(training_data, f)

    print(valid_samples, n_samples)
