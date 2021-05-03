import os
import numpy as np
from joblib import load
window_data = [] 
def check(input_map, input_window, reload_dt):

    dt_map = {}
    global window_data

    # check if data has reached window size
    window_fill_size = len(window_data)
    if window_fill_size < input_window:
        # expand window_data with input_map
        window_data.append(input_map)
        return True

    # FIFO behavior: Buffer of size input_window
    window_data.pop(0)
    window_data.append(input_map)

    # initialize map if not initialized
    if not dt_map:
        for i in range(window_fill_size):
            for key in window_data[i].keys():
                name = f"{key}@{i}"
                dt_map[name] = window_data[i][key]

    #print(dt_map)

    # need to convert to X format
    X = []
    features = ['v@0', 'other_heading@0', 'other_distance@0', 'waypoint_25_dtc@0', 'waypoint_20_dtc@0', 'waypoint_15_dtc@0', 'waypoint_10_dtc@0', 'waypoint_5_dtc@0', 'waypoint_0_dtc@0', 'v@1', 'other_heading@1', 'other_distance@1', 'waypoint_25_dtc@1', 'waypoint_20_dtc@1', 'waypoint_15_dtc@1', 'waypoint_10_dtc@1', 'waypoint_5_dtc@1', 'waypoint_0_dtc@1', 'v@2', 'other_heading@2', 'other_distance@2', 'waypoint_25_dtc@2', 'waypoint_20_dtc@2', 'waypoint_15_dtc@2', 'waypoint_10_dtc@2', 'waypoint_5_dtc@2', 'waypoint_0_dtc@2', 'v@3', 'other_heading@3', 'other_distance@3', 'waypoint_25_dtc@3', 'waypoint_20_dtc@3', 'waypoint_15_dtc@3', 'waypoint_10_dtc@3', 'waypoint_5_dtc@3', 'waypoint_0_dtc@3', 'v@4', 'other_heading@4', 'other_distance@4', 'waypoint_25_dtc@4', 'waypoint_20_dtc@4', 'waypoint_15_dtc@4', 'waypoint_10_dtc@4', 'waypoint_5_dtc@4', 'waypoint_0_dtc@4', 'v@5', 'other_heading@5', 'other_distance@5', 'waypoint_25_dtc@5', 'waypoint_20_dtc@5', 'waypoint_15_dtc@5', 'waypoint_10_dtc@5', 'waypoint_5_dtc@5', 'waypoint_0_dtc@5', 'v@6', 'other_heading@6', 'other_distance@6', 'waypoint_25_dtc@6', 'waypoint_20_dtc@6', 'waypoint_15_dtc@6', 'waypoint_10_dtc@6', 'waypoint_5_dtc@6', 'waypoint_0_dtc@6', 'v@7', 'other_heading@7', 'other_distance@7', 'waypoint_25_dtc@7', 'waypoint_20_dtc@7', 'waypoint_15_dtc@7', 'waypoint_10_dtc@7', 'waypoint_5_dtc@7', 'waypoint_0_dtc@7', 'v@8', 'other_heading@8', 'other_distance@8', 'waypoint_25_dtc@8', 'waypoint_20_dtc@8', 'waypoint_15_dtc@8', 'waypoint_10_dtc@8', 'waypoint_5_dtc@8', 'waypoint_0_dtc@8', 'v@9', 'other_heading@9', 'other_distance@9', 'waypoint_25_dtc@9', 'waypoint_20_dtc@9', 'waypoint_15_dtc@9', 'waypoint_10_dtc@9', 'waypoint_5_dtc@9', 'waypoint_0_dtc@9']
    for feature in features:
        X.append(dt_map[feature])
    X = np.array(X)
    X = np.expand_dims(X, axis=0)
    dts = []
    if not os.path.isdir('./examples/carla/overtake_control/monitor/dt_ego'): return 1
    prev_tree_files = os.listdir("./examples/carla/overtake_control/monitor/dt_ego")
    for fname in prev_tree_files:  # tree_0.joblib
        if fname.endswith("joblib"):
            dts.append(load(f"./examples/carla/overtake_control/monitor/dt_ego/{fname}"))
    # if no tree is found, use AC
    if not dts: return 1

    for dt in dts:
        verdict = dt.predict(X)[0]
        if verdict == 0: return 0
    return 1
