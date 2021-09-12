from torch.utils.data import Dataset
import pickle


class MonitorLearningDataset(Dataset):
    def __init__(self, pkl_file):
        with open() as f:
            data = pickle.load(f)
        self.data = data

    def __len__(self):
        len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        x = np.expand_dims(x, axis=0)
        y = np.expand_dims(y, axis=0)
        return x, y
