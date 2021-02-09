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

# data=readJsonFile('./data_task1_train.json')



# test image 
def showRetangleFromFile(k,v):
	img=cv2.imread(k)
	with open(v,'r') as f:
		for line in f:
			bx=line.split(',')
			x=int(bx[0])
			y=int(bx[1])
			a=int(bx[4])
			b=int(bx[5])
			cv2.rectangle(img, (x, y), (a, b), (0, 255,0), 2)
	plt.imshow(img)
	plt.show()
			

def showRetangle(img,corList):  # corList [[x1_1, y1_1,x3_1,y3_1],...]
	for obj in corList: 
		cv2.rectangle(img, (obj[0], obj[1]), (obj[4], obj[5]), (0, 255,0), 2 )
	plt.imshow(img)
	plt.show()



