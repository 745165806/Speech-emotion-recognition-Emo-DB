import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
import numpy as np


class myNet(nn.Module):
    def __init__(self):
        super(myNet, self).__init__()  # 调用父类的初始化函数
        self.first_conv = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(12, 16),
                                    padding=(11, 15), dilation=(2, 2), stride=(1, 1))
        self.first_maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.second_conv = nn.Conv2d(in_channels=16, out_channels=24, kernel_size=(8, 12),
                                     padding=(7, 11), dilation=(2, 2), stride=(1, 1))
        self.second_maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.third_conv = nn.Conv2d(in_channels=24, out_channels=32, kernel_size=(5, 7),
                                     padding=(2, 3), dilation=(1, 1), stride=(1, 1))
        self.third_maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.lstm_layer = nn.LSTM(input_size=925, hidden_size=128, num_layers=1,
                                  bidirectional=True, batch_first= True)

        self.first_dense = nn.Linear(256,4)#64
        self.first_ReLu = nn.ReLU()
        # self.dropout_layer = nn.Dropout(p=0.2)
        # self.second_dense = nn.Linear(64,4)
        # self.second_ReLu = nn.ReLU()
        self.softmax_layer = nn.Softmax(dim=1)



    def forward(self, x):
        x = self.first_conv(x)
        x = self.first_maxpool(x)
        x = self.second_conv(x)
        x = self.second_maxpool(x)
        x = self.third_conv(x)
        x = self.third_maxpool(x)
        batch_size, channel_dim, fre_dim , time_dim = x.size()
        x = x.view(batch_size, channel_dim, -1)
        x, _ = self.lstm_layer(x)
        x = x[:, -1, :]
        x = self.first_dense(x)
        x = self.first_ReLu(x)
        # x = self.dropout_layer(x)
        # x = self.second_dense(x)
        # x = self.second_ReLu(x)
        x = self.softmax_layer(x)
        return x

if __name__ == '__main__':
    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    model = myNet()
    if torch.cuda.is_available():
        model.cuda()
    print(model)
    print("Start")
    summary(model, (1, 200, 300), device='cuda')
