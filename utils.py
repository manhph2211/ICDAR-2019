import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import glob
import os


data_train_path='./data/task1_train'
image_path=glob.glob(os.path.join(data_train_path,'*.jpg'))

img=cv2.imread(image_path[0])
print(img.shape)