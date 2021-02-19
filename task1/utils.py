import glob
import os
import re
import json
import cv2
import matplotlib.pyplot as plt 

# image_path=glob.glob(os.path.join(data_train_path,'*.jpg'))
# txt_path=glob.glob(os.path.join(data_train_path,'*.txt'))

def get_data(data_train_path='../data/task1_train'):
	new_data=[]
	data={}
	for file in os.listdir(data_train_path):
		if re.match('^((?!\)).)*$',file):
			new_data.append(re.sub('.txt|.jpg$','',file))
	codeSet=set(new_data)

	for code in codeSet:
		data[os.path.join(data_train_path,code+'.jpg')]=os.path.join(data_train_path,code+'.txt')

	return data



def save_jsonFile(data):
	with open('./data_task1_train.json','w') as f:
		json.dump(data,f,indent=4)

def read_json_file(path='./data_task1_train.json'):
	with open(path,'r') as f:
		data=json.load(f)
		return data


def show_retangle_from_file(k,v):
	img=cv2.imread(k)
	with open(v,'r') as f:
		for line in f:
			bx=line.split(',')
			x_min=int(bx[0])
			y_min=int(bx[1])
			x_max=int(bx[4])
			y_max=int(bx[5])
			cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255,0), 2)
	plt.imshow(img)
	plt.show()


if __name__=="__main__":
	#data=get_data()
	#save_jsonFile(data)
	data=read_json_file()
	item=iter(data.items())
	next_item=next(item)
	show_retangle_from_file(next_item[0],next_item[1])







