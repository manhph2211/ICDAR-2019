import torch
from torch.utils.data import Dataset,DataLoader
import json
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.preprocessing import MinMaxScaler


class my_dataset(Dataset):

    def __init__(self,img_paths,targets_paths,phase):
        self.img_paths=img_paths
        self.targets=targets_paths
        self.transforms=transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.phase=phase


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self,idx):

        img=cv2.imread(self.img_paths[idx]) # BGR
        height,width,channels=img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
        img=torch.tensor(img, dtype=torch.float32)
        img=img.permute(2, 0, 1) # (height, width, channels) -> (channels, height, width)
        img=self.transforms(img)


        tar_path=self.targets_paths[idx]

        with open(tar_path,'r') as f:
            boxes=[]
            labels=[]
            for line in f:
                bx=line.split(',')
                x_min=int(bx[0])
                y_min=int(bx[1])
                x_max=int(bx[4])
                y_max=int(bx[5])
                boxes.append([x_min,y_min,x_max,y_max])
                labels.append(1)
            

            mms = MinMaxScaler().fit(boxes)
            boxes_norm=mms.transform(boxes)
            boxes=torch.tensor(boxes,dtype=torch.long)
            labels=torch.tensor(labels,dtype=torch.long)
            tar=torch.hstack((boxes,labels))
            return img,tar 
    

def read_son_file(path):
	with open(path,'r') as f:
		data=json.load(f)
		return data


if __name__=="__main__":
    pass


