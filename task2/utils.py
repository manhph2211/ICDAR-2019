import cv2
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from task1.utils import read_json_file, show_retangle_from_file


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
			tex=''.join(bx[8:])
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





if __name__ == '__main__':
	data=read_json_file()
	item=next(iter(data.items()))
	test_img_path=item[0]
	test_txt_path=item[1]
	crop_img,text=get_text_img(test_img_path,test_txt_path)
	test_crop_img=crop_img[0]
	test_crop_text=text[0]
	plt.figure(figsize=(15, 15))
	plt.imshow(test_crop_img)
	plt.show()

	# crop_list, text_list, charac_list=get_data()

	# print(len(charac_list))
	# print(charac_list)