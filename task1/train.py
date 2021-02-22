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

batch_size = 32
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=my_collate_fn)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle=False, collate_fn=my_collate_fn)
dataloader_dict = {"train": train_dataloader, "val": val_dataloader}

# network
cfg = {
    "num_classes": 2,
    "input_size": 300, #SSD300
    "bbox_aspect_num": [4, 6, 6, 6, 4, 4], # Tỷ lệ khung hình cho source1->source6
    "feature_maps": [38, 19, 10, 5, 3, 1],
    "steps": [8, 16, 32, 64, 100, 300], # Size of default box
    "min_size": [30, 60, 111, 162, 213, 264], # Size of default box
    "max_size": [60, 111, 162, 213, 264, 315], # Size of default box
    "aspect_ratios": [[2], [2,3], [2,3], [2,3], [2], [2]]
}

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
optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

# training, validation
def train_model(net, dataloader_dict, criterion, optimizer, num_epochs):
    # move network to GPU
    net.to(device)

    iteration = 1
    epoch_train_loss = 0.0
    epoch_val_loss = 0.0
    best_val_loss=99999
    logs = []
    for epoch in range(num_epochs+1):
      
        for phase in ["train", "val"]:
            if phase == "train":
                net.train()
                print("(Training)")
            else:
                if (epoch+1) % 10 == 0:
                    net.eval() 
                    print("---"*10)
                    print("(Validation)")
                else:
                    continue
            for images, targets in dataloader_dict[phase]:
                # move to GPU or not!
                images = images.to(device)
                targets = [ann.to(device) for ann in targets]

                # init optimizer
                optimizer.zero_grad()
                # forward
                with torch.set_grad_enabled(phase=="train"):
                    outputs = net(images)
                    loss_l, loss_c = criterion(outputs, targets)
                    loss = loss_l + loss_c # alpha=1

                    if phase == "train":
                        loss.backward() # calculate gradient
                        nn.utils.clip_grad_value_(net.parameters(), clip_value=2.0)
                        optimizer.step() # update parameters

                        if (iteration % 10) == 0:
                            print("Iteration {} || Loss: {:.4f}".format(iteration, loss.item() ))
                        epoch_train_loss += loss.item()
                        iteration += 1
                    else:
                        epoch_val_loss += loss.item()
        print("---"*20)
        print("Epoch {} || epoch_train_loss: {:.4f} || Epoch_val_loss: {:.4f}".format(epoch+1, epoch_train_loss, epoch_val_loss))           
        log_epoch = {"epoch": epoch+1, "train_loss": epoch_train_loss, "val_loss": epoch_val_loss}
        logs.append(log_epoch)
        df = pd.DataFrame(logs)
        df.to_csv("./weights/ssd_logs.csv")
        torch.save(net.state_dict(), "./weights/ssd300" + ".pth")

        if epoch_val_loss<best_val_loss:
            best_val_loss=epoch_val_loss
            torch.save(net.state_dict(), "./weights/ssd300_best_val" + ".pth")
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0

num_epochs = 100
train_model(net, dataloader_dict, criterion, optimizer, num_epochs=num_epochs)
