import glob
import os
import re
import json
import cv2
import matplotlib.pyplot as plt 

# image_path=glob.glob(os.path.join(data_train_path,'*.jpg'))
# txt_path=glob.glob(os.path.join(data_train_path,'*.txt'))

def getData(data_train_path='../data/task1_train'):
	new_data=[]
	data={}
	for file in os.listdir(data_train_path):
		if re.match('^((?!\)).)*$',file):
			new_data.append(file.strip('.txt|.jpg'))
	codeSet=set(new_data)

	for code in codeSet:
		data[os.path.join(data_train_path,code+'.jpg')]=os.path.join(data_train_path,code+'.txt')

	return data


def saveJsonFile(data):
	with open('./data_task1_train.json','w') as f:
		json.dump(data,f,indent=4)


# data=getData()
# saveJsonFile(data)


# test image 
# data=readJsonFile('./data_task1_train.json')
'''
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
			
'''


def test(img,corList):  # corList [[x1_1, y1_1,x3_1,y3_1],...]
	for obj in corList: 
		cv2.rectangle(img, (obj[0], obj[1]), (obj[4], obj[5]), (0, 255,0), 2 )
	plt.imshow(img)
	plt.show()

