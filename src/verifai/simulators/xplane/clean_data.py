import numpy as np
import pandas as pd
import pickle
import os

input_window = 5
decision_window = 2
horizon = 2
sample_rate = 1

# output_file = "training_data"

# log = open(f"{output_file}.csv", "w")
# log.write("sample, input_window, prediction\n")

sample_count = 0
valid_samples = 0
train_data = []
for file in os.listdir('./simulation_data'):
	if file.endswith('.csv'):
		filename = './simulation_data/' + file
		print(filename)
		data = pd.read_csv(filename, dtype='str')

		times = [float(t) for t in data['time'].values]
		lat = [float(l) for l in data[' lat'].values]
		lon = [float(l) for l in data[' lon'].values]
		cte = [float(c) for c in data[' cte'].values]
		he = [float(h) for h in data[' he'].values]
		dist = [float(d) for d in data[' pos'].values]

		# static data
		init_h = float(data[' init_heading'].values[0]) 
		init_ct = float(data[' init_pos'].values[0])
		tod = float(data[' day_time'].values[0])
		clouds = float(data[' clouds'].values[0])
		rain = float(data[' rain'].values[0])

		indices_to_keep = [0]
		prev_time = times[0]
		for i, time in enumerate(times[1:]):
			if time - prev_time > sample_rate:
				indices_to_keep.append(i+1) 
				prev_time = time

		times = [times[i] for i in indices_to_keep]
		lat = [lat[i] for i in indices_to_keep]
		lon = [lon[i] for i in indices_to_keep]
		cte = [cte[i] for i in indices_to_keep]
		he = [he[i] for i in indices_to_keep]
		dist = [dist[i] for i in indices_to_keep]

		assert len(times) == len(lat) == len(lon) == len(cte) == len(he) == len(dist)

		for i in range(len(times) - (decision_window + horizon + input_window)):
			concat_data = []
			for j in range(i, i + input_window):
				concat_data.append((tod, clouds, rain, init_h, init_ct, lat[j], lon[j], cte[j], he[j], dist[j]))
			ind = i+horizon+input_window
			# TODO: write this into a function
			prediction = all([c < 2 for c in cte[ind:ind+decision_window]])
			train_data.append((concat_data, prediction))
			valid_samples += int(prediction)
			# log.write(f"{sample_count}, {concat_data}, {prediction}\n")
			sample_count += 1

# log.close()
with open('training_data.pkl', 'wb') as f:
	pickle.dump(train_data, f)


print(valid_samples)