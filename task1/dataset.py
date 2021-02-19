import torch
from torch.utils.data import Dataset
import cv2
from torchvision import transforms
from sklearn.preprocessing import MinMaxScaler
from utils import read_json_file
from tqdm import tqdm



class my_dataset(Dataset):

    def __init__(self,img_paths,targets_paths):
        self.img_paths=img_paths
        self.targets_paths=targets_paths
        self.transforms=transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self,idx):

        img=cv2.imread(self.img_paths[idx]) # BGR
        #height,width,channels=img.shape
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
                labels.append([1])
            # minmax norm:v
            mms = MinMaxScaler().fit(boxes)
            boxes_norm=mms.transform(boxes)
            boxes=torch.tensor(boxes_norm,dtype=torch.long)
            labels=torch.tensor(labels,dtype=torch.long)
            tar=torch.hstack((boxes,labels))
            return img,tar
    
if __name__=="__main__":
    data=read_json_file()
    img_paths=list(data.keys())
    targets_paths=list(data.values())
    my_dataset_=my_dataset(img_paths,targets_paths)
    my_loader = torch.utils.data.DataLoader(
        my_dataset_,
        batch_size=8,
        num_workers=8,
        shuffle=True,
    )

    for k,v in tqdm(my_loader,total=len(my_loader)):
        print(v.shape)



