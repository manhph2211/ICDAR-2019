import torch
import cv2
from torchvision import transforms
from torch.utils.data import Dataset

class my_dataset(Dataset):
  
  def __init__(self,img_paths,txt_paths):
    self.img_paths=img_paths
    self.txt_paths=txt_paths
   
  def __getitem__(self,idx):
    img=cv2.imread(self.img_paths[idx])
    img=cv2.resize(img,(100,32)) 
    img=torch.tensor(img, dtype=torch.float32)
    img=img.permute(2, 0, 1)
    img_transform=transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img=img_transform(img)
    txt=self.txt_paths[idx]
    txt=read_text_file(txt)
    txt=encode(txt)
    txt=torch.IntTensor(txt)
    return img,txt

  def __len__(self):
    return len(self.img_paths)


def my_collate_fn(batch):
  images = list()
  texts = list()
  for b in batch:
    images.append(b[0])
    texts.append(b[1])
  imgs = torch.cat([t.unsqueeze(0) for t in images], 0) 
  return imgs, texts

