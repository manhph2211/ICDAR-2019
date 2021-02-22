# dataloader
# network -> SSD300
# loss -> MultiBoxLoss
# optimizer
# training, validation
import torch
import torch.nn as nn
from dataset import my_dataset, my_collate_fn
from model import SSD
from multiboxloss import MultiBoxLoss
from utils import read_json_file
from sklearn import model_selection
import pandas as pd
from config import cfg


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)
torch.backends.cudnn.benchmark = True

# dataloader
data=read_json_file()
img_paths = list(data.keys())
targets_paths = list(data.values())
## split
X_train, X_val, y_train, y_val = model_selection.train_test_split(img_paths, targets_paths, test_size=0.2, random_state=1)

train_dataset = my_dataset(X_train, y_train)
val_dataset = my_dataset(X_val,y_val)

batch_size = 16
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=my_collate_fn)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle=False, collate_fn=my_collate_fn)
dataloader_dict = {"train": train_dataloader, "val": val_dataloader}

# network

net = SSD(phase="train", cfg=cfg)

#pretrain
vgg_weights = torch.load("./pretrained/vgg16_reducedfc.pth")
net.vgg.load_state_dict(vgg_weights)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

# He init
net.extras.apply(weights_init)
net.loc.apply(weights_init)
net.conf.apply(weights_init)

# MultiBoxLoss
criterion = MultiBoxLoss(jaccard_threshold=0.5, neg_pos=3, device=device)

# optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=3e-4, weight_decay=5e-4)



# training, validation
def train_model(net, dataloader_dict, criterion, optimizer, num_epochs):
    # move network to GPU
    net.to(device)
    epoch_train_loss = 0.0
    epoch_val_loss = 0.0
    val_loss=[]
    logs = []
    best_val_loss=999999
    print("Training...")
    for epoch in range(num_epochs):
        for phase in ["train", "val"]:
            if phase == "train":
              net.train()
    
              for images, targets in dataloader_dict['train']:
                  # move to GPU or not!
                  images = images.to(device)
                  targets = [ann.to(device) for ann in targets]

                  # init optimizer
                  optimizer.zero_grad()
                  # forward
                  with torch.set_grad_enabled(phase=="train"):
                    outputs = net(images)
                    loss_l, loss_c = criterion(outputs, targets)
                    loss = loss_l + loss_c 
                    loss.backward() # calculate gradient
                    nn.utils.clip_grad_value_(net.parameters(), clip_value=2.0)
                    optimizer.step() # update parameters
                    epoch_train_loss += loss.item()
                    
            else:
              net.eval()
             
              for images, targets in dataloader_dict['val']:
                images = images.to(device)
                targets = [ann.to(device) for ann in targets]
                outputs = net(images)
                loss_l, loss_c = criterion(outputs, targets)
                loss = loss_l + loss_c # alpha=1
                epoch_val_loss += loss.item()
        print("---"*20)
        print("Epoch {} || epoch_train_loss: {:.4f} || Epoch_val_loss: {:.4f}".format(epoch+1, epoch_train_loss, epoch_val_loss))           
        log_epoch = {"epoch": epoch+1, "train_loss": epoch_train_loss, "val_loss": epoch_val_loss}
        logs.append(log_epoch)
        val_loss.append(epoch_val_loss)
        df = pd.DataFrame(logs)
        df.to_csv("./data/ssd_logs.csv")
        torch.save(net.state_dict(), "./weights/ssd300_" + ".pth")
        if epoch_val_loss<best_val_loss:
            best_val_loss=epoch_val_loss
            torch.save(net.state_dict(), "./weights/ssd300_best_val" + ".pth")
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0
            


num_epochs = 50
train_model(net, dataloader_dict, criterion, optimizer, num_epochs=num_epochs)
