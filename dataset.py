from torch.utils.data import Dataset
import numpy as np

class CustomPlaces(Dataset):
    def __init__(self, data: list[np.ndarray], transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        img, mask = self.transform(img)
        return img, mask
