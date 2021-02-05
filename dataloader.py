import torch
from torch.utils.data import Dataset,DataLoader
import json
import cv2
import matplotlib.pyplot as plt

class makeDataset(Dataset):
    def __init__(self,data,labels):
        self.data=data
        self.labels=labels
        self.n_samples=data.shape[0]
        
    def __getitem__(self,idx):
        return self.data[idx],self.labels[idx]

    
    def __len__(self):
        return len(self.n_samples)
    

def readJsonFile(path):
	with open(path,'r') as f:
		data=json.load(f)
		return data

#x1_1, y1_1,x2_1,y2_1,x3_1,y3_1,x4_1,y4_1
def getRetangle(image,line):
	bx=line.split(',')
	x=int(bx[0])
	y=int(bx[1])
	a=int(bx[2])
	b=int(bx[3])

	cv2.rectangle(image, (x, y), (a, b), (0, 255,0), 2 )


data=readJsonFile('./data_task1_train.json')

k=list(data.keys())[0]
v=list(data.values())[0]

def showRetangle(k,v):
	img=cv2.imread(k)
	with open(v,'r') as f:
		for line in f:
			getRetangle(img,line)
	plt.plot(img)

	
showRetangle(k,v)
plt.show()