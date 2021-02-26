
import torch.nn as nn
from utils import split_data,read_json_file, get_text
from dataset import my_dataset,alignCollate
from model import my_model
from engine import train_fn,eval_fn
import torch
import cv2

data=read_json_file(path='../data/For_task_2/data.json')
img_paths=list(data.keys())
txt_paths=list(data.values())

batch_size=32
X_train,y_train,X_val,y_val,X_test,y_test=split_data(img_paths,txt_paths)

train_dataset = my_dataset(X_train,X_train)
val_dataset = my_dataset(X_val,y_val)
test_dataset = my_dataset(X_test,y_test)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=alignCollate,)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle=False, collate_fn=alignCollate,)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False, collate_fn=alignCollate,)



model=my_model(69)
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

model.apply(weights_init)


NUM_EPOCH=5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

for epoch in range(NUM_EPOCH):
	train_loss = train_fn(model, train_dataloader, optimizer,device)
	test_loss = eval_fn(model, val_dataloader,device)
	print("Epoch={0} Train Loss={1}, Val Loss={2}".format(epoch+1,train_loss,test_loss))