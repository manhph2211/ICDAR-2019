import torch
from torch.utils.data import Dataset,DataLoader
import json
import cv2
import matplotlib.pyplot as plt



class makeDataset(Dataset):

    def __init__(self,json_file):
        pass
        
    def __getitem__(self,idx):
        pass

    
    def __len__(self):
        pass
    

def readJsonFile(path):
	with open(path,'r') as f:
		data=json.load(f)
		return data


if __name__=="__main__":
    print(len(readJsonFile('./data_task1_train.json')))


