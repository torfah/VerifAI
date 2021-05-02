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
    features = ['v@0', 'waypoint_15_dtc@0', 'waypoint_12_dtc@0', 'waypoint_9_dtc@0', 'waypoint_6_dtc@0', 'waypoint_3_dtc@0', 'waypoint_0_dtc@0', 'v@1', 'waypoint_15_dtc@1', 'waypoint_12_dtc@1', 'waypoint_9_dtc@1', 'waypoint_6_dtc@1', 'waypoint_3_dtc@1', 'waypoint_0_dtc@1', 'v@2', 'waypoint_15_dtc@2', 'waypoint_12_dtc@2', 'waypoint_9_dtc@2', 'waypoint_6_dtc@2', 'waypoint_3_dtc@2', 'waypoint_0_dtc@2', 'v@3', 'waypoint_15_dtc@3', 'waypoint_12_dtc@3', 'waypoint_9_dtc@3', 'waypoint_6_dtc@3', 'waypoint_3_dtc@3', 'waypoint_0_dtc@3', 'v@4', 'waypoint_15_dtc@4', 'waypoint_12_dtc@4', 'waypoint_9_dtc@4', 'waypoint_6_dtc@4', 'waypoint_3_dtc@4', 'waypoint_0_dtc@4', 'v@5', 'waypoint_15_dtc@5', 'waypoint_12_dtc@5', 'waypoint_9_dtc@5', 'waypoint_6_dtc@5', 'waypoint_3_dtc@5', 'waypoint_0_dtc@5', 'v@6', 'waypoint_15_dtc@6', 'waypoint_12_dtc@6', 'waypoint_9_dtc@6', 'waypoint_6_dtc@6', 'waypoint_3_dtc@6', 'waypoint_0_dtc@6', 'v@7', 'waypoint_15_dtc@7', 'waypoint_12_dtc@7', 'waypoint_9_dtc@7', 'waypoint_6_dtc@7', 'waypoint_3_dtc@7', 'waypoint_0_dtc@7', 'v@8', 'waypoint_15_dtc@8', 'waypoint_12_dtc@8', 'waypoint_9_dtc@8', 'waypoint_6_dtc@8', 'waypoint_3_dtc@8', 'waypoint_0_dtc@8', 'v@9', 'waypoint_15_dtc@9', 'waypoint_12_dtc@9', 'waypoint_9_dtc@9', 'waypoint_6_dtc@9', 'waypoint_3_dtc@9', 'waypoint_0_dtc@9']
    for feature in features:
        X.append(dt_map[feature])
    X = np.array(X)
    X = np.expand_dims(X, axis=0)
    dts = []
    prev_tree_files = os.listdir(f"./examples/carla/overtake_control/monitor/dt_other")
    for fname in prev_tree_files:  # tree_0.joblib
        if fname.endswith("joblib"):
            dts.append(load(f"./examples/carla/overtake_control/monitor/dt_other/{fname}"))

    # if no tree is found, use AC
    if not dts: return 1

    a = 0.1
    v_sum = 0
    for i in range(len(dts)):
        verdict = dts[i].predict(X)[0]
        v_sum += ( a**(float(i!=0)) )* ( (1-a)**(len(dts)-1-i) ) * float(verdict)
    if v_sum >= 0.5:
        return 1
    else:
        return 0
