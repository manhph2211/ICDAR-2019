import sys
sys.path.append("..")
from task1.utils import read_json_file
from sklearn import model_selection
import os
import json
import cv2
import matplotlib.pyplot as plt


def save_json_file(data,path):
	with open(path,'w') as f:
		json.dump(data,f,indent=4)


def get_text_img(img_path,txt_path):

	img = cv2.imread(img_path)
	text_img=[]
	text=[]
	img_list=[]
	with open(txt_path,'r') as f:
		for line in f:
			bx=line.split(",")
			x_min=int(bx[0])
			y_min=int(bx[1])
			x_max=int(bx[4])
			y_max=int(bx[5])
			crop_img = img[y_min:y_max, x_min:x_max,:]
			img_list.append(crop_img)
			tex=','.join(bx[8:])
			text.append(tex)

	return img_list,text


def get_data():
	data=read_json_file()
	crop_list=[]
	text_list=[]
	charac_list=[]
	for img_path,txt_path in data.items():
		img=cv2.imread(img_path)

		with open(txt_path, 'r') as f:
			for line in f:

				bx = line.split(",")
				x_min = int(bx[0])
				y_min = int(bx[1])
				x_max = int(bx[4])
				y_max = int(bx[5])
				crop_img = img[y_min:y_max, x_min:x_max,:]
				crop_list.append(crop_img)
				text=''.join(bx[8:])
				for cha in text:
					charac_list.append(cha)
				text_list.append(text)

	return crop_list,text_list,charac_list


def split_data(img_paths,targets):
	X_train, X_test, y_train, y_test = model_selection.train_test_split(img_paths, targets, test_size=0.2, random_state=1)
	X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2
	return X_train,y_train,X_val,y_val,X_test,y_test



def create_data_folder(data):   # data -- dic
	k=1
	for i,(img_path,txt_path) in enumerate(data.items()):
	  
	  crop_img_list,text=get_text_img(img_path,txt_path)
	  assert len(crop_img_list)==len(text)
	  for j in range(len(crop_img_list)):

	    save_img_path='../data/For_task_2/Crop_Img/'+"%d"%(k)+'.jpg'
	    if not os.path.exists(save_img_path):
	      cv2.imwrite(save_img_path,crop_img_list[j])
	    
	    save_txt_path="../data/For_task_2/Text/"+'%d'%(k)+'.txt'
	    if not os.path.exists(save_txt_path):
	      with open(save_txt_path,'w') as f:
	        f.write(text[j])
	    k+=1


def get_paths(img_dir='../data/For_task_2/Crop_Img',txt_dir='../data/For_task_2/Text'):
  dic={}
  for i in range(1,33627):
    img_name='%d.jpg'%i
    txt_name='%d.txt'%i
    
    img_path=os.path.join(img_dir,img_name)
    txt_path=os.path.join(txt_dir,txt_name)
    if not os.path.exists(img_path) or not os.path.exists(img_path):
      continue
    
    dic[img_path]=txt_path

  return dic


def get_text(txt_path):
  with open(txt_path,'r') as f:
    tx=f.read()
    return tx.strip()


def get_vocab(txt_paths):
	vocab=[]
	for path in txt_paths:
		text=get_text(path)
		for cha in text:
			if cha not in vocab:
				vocab.append(cha)

	return "".join(vocab)



if __name__ == '__main__':
	data=read_json_file()
	create_data_folder(data)
	dic=get_paths()
	save_json_file(dic,'../data/For_task_2/data.json')