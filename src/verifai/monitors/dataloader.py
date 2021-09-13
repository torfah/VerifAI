from torch.utils.data import Dataset
import numpy as np
import pickle


class MonitorLearningDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        trace, satisfy = self.data[idx]

        x = []
        next_params = []
        for param, value in trace:
            if param == 'time':
                if len(next_params) > 0:
                    x.append(next_params)
                next_params = []
                continue
            if type(value) == tuple: # keep this for now
                next_params.append(float(value[0])) 
            else: 
                next_params.append(float(value))

        x.append(next_params)

        x = np.array(x)
        y = float(int(satisfy))       
        return x, y
