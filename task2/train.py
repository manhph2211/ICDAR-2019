import torch
import torch.nn as nn
from utils import split_data,read_json_file, get_text
from dataset import my_dataset,my_collate_fn
from model import my_model,weights_init
from engine import train_fn,eval_fn
import cv2
from sklearn import model_selection
import pandas as pd


vocab="- !#$%&'()*+,./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`lr{|}~\""
num_cha=len(vocab)
print(num_cha)
data=read_json_file(path='../data/For_task_2/data.json')
img_paths=list(data.keys())
txt_paths=list(data.values())

batch_size=32
X_train, X_val, y_train, y_val = model_selection.train_test_split(img_paths, txt_paths, test_size=0.2, random_state=1)

train_dataset = my_dataset(X_train,y_train,vocab)
val_dataset = my_dataset(X_val,y_val,vocab)
#test_dataset = my_dataset(X_test,y_test)

print(len(train_dataset))

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=my_collate_fn,)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle=False, collate_fn=my_collate_fn,)
#test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False,collate_fn=my_collate_fn,)


model=my_model(num_cha)
model.apply(weights_init)
NUM_EPOCHS=50
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using ",device)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
MODEL_SAVE_PATH = './weights/my_model.pth'
# model.load_state_dict(torch.load(MODEL_SAVE_PATH))

def train(model,MODEL_SAVE_PATH ,NUM_EPOCHS,optimizer):
	best_val_loss=999
	print("Training...")
	log=[]
	for epoch in range(1,NUM_EPOCHS+1):
	    train_loss = train_fn(model, train_dataloader, optimizer,device)
	    val_loss = eval_fn(model, val_dataloader,device)

	    log_epoch = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
	    log.append(log_epoch)
	    df = pd.DataFrame(log)
	    df.to_csv("./weights/logs2.csv")   
	    if val_loss < best_val_loss:
	        best_val_loss = val_loss
	        torch.save(model.state_dict(),MODEL_SAVE_PATH)
	    print("Epoch {} || epoch_train_loss: {:.4f} || Epoch_val_loss: {:.4f} ".format(epoch + 1,train_loss, val_loss))


train(model,MODEL_SAVE_PATH ,NUM_EPOCHS,optimizer)