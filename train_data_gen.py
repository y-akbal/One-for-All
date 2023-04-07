import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class CSVDataset(Dataset):
    def __init__(self, csv_dir, transform=None):
        self.csv_dir = csv_dir
        self.transform = transform
        self.files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        csv_path = os.path.join(self.csv_dir, self.files[idx])
        data = pd.read_csv(csv_path)
        if self.transform:
            data = self.transform(data)
        return data.values

class BatchCSVDataLoader:
    def __init__(self, csv_dir, batch_size, shuffle=True, num_workers=0):
        self.dataset = CSVDataset(csv_dir)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    def __iter__(self):
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size,
                                shuffle=self.shuffle, num_workers=self.num_workers)
        for batch in dataloader:
            yield batch

    def __len__(self):
        return len(self.dataset) // self.batch_size


dataloader = BatchCSVDataLoader('path/to/csv/dir', batch_size=32, shuffle=True, num_workers=4)
for batch in dataloader:
    # process batch
    continue
