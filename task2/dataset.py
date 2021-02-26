# tough one


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
    #img=cv2.resize(img,(100,32)) 
    img=torch.tensor(img, dtype=torch.float32)
    img=img.permute(2, 0, 1)
    txt=get_text(self.txt_paths[idx])
    txt=encode(txt)
    txt=torch.IntTensor(txt)
    return img,txt

  def __len__(self):
    return len(self.img_paths)


# def my_collate_fn(batch):
#     imgs, labels = zip(*batch)
#     #imgs = torch.stack(imgs, dim=0)
#     imgs = torch.cat([t.unsqueeze(0) for t in imgs], 0) 
#     return imgs, labels

class resizeNormalize(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        img = cv2.resize(img,(self.size,self.size))
        img_transform=transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        img=transform(img)
        return img


class alignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, batch):
        images, labels = zip(*batch)

        imgH = self.imgH
        imgW = self.imgW
        if self.keep_ratio:
            ratios = []
            for image in images:
                w, h = image.size
                ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1]
            imgW = int(np.floor(max_ratio * imgH))
            imgW = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW

        transform = resizeNormalize((imgW, imgH))
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        return images, labels

