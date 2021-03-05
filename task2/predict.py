import cv2
from torchvision import transforms
from dataset import decode
from model import my_model
import torch
import matplotlib.pyplot as plt
import glob 
import numpy as np



def predict(img_path,model,vocab):
	img=cv2.imread(img_path)
	plt.imshow(img)
	img=cv2.resize(img,(100,32)) 
	img=torch.tensor(img, dtype=torch.float32)
	img=img.permute(2, 0, 1)
	img=img/255
	img=transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])(img)
	img=img.reshape(1,img.shape[0],img.shape[1],img.shape[2])
	pred=model(img)
	pred=pred.permute(1,0,2)
	pred=torch.softmax(pred,2)
	pred=torch.argmax(pred,2)
	pred=pred.detach().cpu().numpy()
	print(decode(pred,vocab))
	plt.show()
	# return pred


if __name__ == '__main__':

	vocab="- !#$%&'()*+,./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`lr{|}~\""
	model=my_model(len(vocab))
	MODEL_SAVE_PATH = './weights/my_model.pth'
	model.load_state_dict(torch.load(MODEL_SAVE_PATH))
	test_dir='../data/For_task_2/Crop_Img/'
	list_path=glob.glob(test_dir+"*.jpg")
	rand_idx=np.random.randint(200)
	predict(list_path[rand_idx],model,vocab)




