from torch.nn import functional as F
import torch.nn as nn


class my_model(nn.Module):
    def __init__(self, num_chars):
        super(my_model, self).__init__()
        self.conv_1 = nn.Conv2d(3, 128, kernel_size=(3, 6), padding=(1, 1))
        self.conv_2 = nn.Conv2d(128, 64, kernel_size=(3, 6), padding=(1, 1))
        self.conv_3 = nn.Conv2d(64,64,kernel_size=(3, 6), padding=(1, 1))
        self.linear_1 = nn.Linear(2048, 64)
        self.drop_1 = nn.Dropout(0.2)
        self.lstm = nn.GRU(64, 32, bidirectional=True, num_layers=2, dropout=0.25, batch_first=True)
        self.output = nn.Linear(64, num_chars)


    def forward(self, images):    
        bs, _, _, _ = images.size()   
        x = F.relu(self.conv_1(images)) 
        x = F.relu(self.conv_2(x))
        x = F.relu(self.conv_3(x))
        x = x.permute(0, 3, 1, 2)
        x = x.view(bs, x.size(1), -1)
        x = F.relu(self.linear_1(x))
        x = self.drop_1(x)
        x, _ = self.lstm(x)
        x = self.output(x)
        x = x.permute(1, 0, 2)
        
        return x



def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)