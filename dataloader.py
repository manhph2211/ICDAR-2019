import torch
from torch.utils.data import Dataset,DataLoader


class makeDataset(Dataset):
    def __init__(self,data,labels):
        self.data=data
        self.labels=labels
        self.n_samples=data.shape[0]
        
    def __getitem__(self,idx):
        return self.data[idx],self.labels[idx]

    
    def __len__(self):
        return len(self.n_samples)
    
