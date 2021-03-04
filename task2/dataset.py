import torch
import cv2
from torchvision import transforms
from torch.utils.data import Dataset
from utils import get_text



def encode(text,vocab):
  encode_text=[]
  for cha in text:
    if cha=="·":
      cha='.'
    if cha not in vocab:
      print(cha)
    idx=vocab.index(cha)+1
    encode_text.append(idx)

  return encode_text


def decode(y_hat,vocab):
  y_hat=F.log_softmax(y_hat, 2) 
  y_hat=y_hat.argmax(2).numpy()
  y_hat=y_hat.T
  result=[]
  for sample in y_hat:
    text=[]
    for idx in sample:
      text.append(idx)
    result.append(text)

  return result


class my_dataset(Dataset):
  
  def __init__(self,img_paths,txt_paths,vocab):
    self.img_paths=img_paths
    self.txt_paths=txt_paths
    self.vocab=vocab
   
  def __getitem__(self,idx):
    img=cv2.imread(self.img_paths[idx])
    img=cv2.resize(img,(100,32)) 
    img=torch.tensor(img, dtype=torch.float32)
    img=img.permute(2, 0, 1)
    img=img/255
    img_transform=transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img=img_transform(img)
    txt=self.txt_paths[idx]
    txt=get_text(txt)
    txt=encode(txt,self.vocab)
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



if __name__ == '__main__':
  main()