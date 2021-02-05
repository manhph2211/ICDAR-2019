import glob
import os
import re
import json
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