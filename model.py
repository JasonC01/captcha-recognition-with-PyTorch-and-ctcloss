import pandas as pd
import collections
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from train import DEVICE
from utils import get_length

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride, padding: int, kernel_size):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, stride=stride, padding=padding, kernel_size=kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor):
        return self.bn(self.conv(x))

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride, kernel_size) -> None:
        super().__init__()
        self.conv1 = ConvBlock(in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=kernel_size, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = ConvBlock(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), stride=1, padding=1)
        self.downsample = ConvBlock(in_channels=in_channels, out_channels=out_channels, stride=stride, padding=1, kernel_size=kernel_size)

    def forward(self, x):
        identity = self.downsample(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x + identity)
        
        return x
    
class CRNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CRNN, self).__init__()

        self.res1 = ResidualBlock(in_channels=in_channels, out_channels=256, kernel_size=(3, 3), stride=1)
        self.res2 = ResidualBlock(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=1)
        self.res3 = ResidualBlock(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1)
        

        self.c1 = nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=(4, 7), stride=1, padding=1)
        self.c2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(4, 7), stride=2, padding=1)

        self.relu = nn.ReLU()

        self.b1 = nn.BatchNorm2d(256)
        self.b2 = nn.BatchNorm2d(256)

        self.cnn = nn.Sequential(
            self.c1,
            self.relu,
            self.b1,
            nn.MaxPool2d(3, 3),
            self.c2,
            self.relu,
            self.b2
        )

        self.rcnn = nn.Sequential(
            self.res1,
            nn.MaxPool2d(3, 3),
            self.res2,
            nn.MaxPool2d(3, 3),
            self.res3,
            nn.MaxPool2d(3, 3)
        )

        # self.fc1 = nn.Linear(1024, 512)

        self.rnn = nn.LSTM(1024, 512, bidirectional=True, batch_first=False)

        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # print(x.shape)
        x = self.rcnn(x)
        # print(x.shape)
        N, C, h, w = x.size()
        x = x.view(N, -1, w)
        x = x.permute(2, 0, 1)
        # print(x.shape)
        #x = x.view(N, -1, h)
        #x = x.permute(0, 2, 1)
        # x = self.fc1(x)
        # print(x.shape)
        #x = x.permute(1, 0, 2)
        x, _ = self.rnn(x)
        #print(x.shape)
        x = self.fc2(x)
        # print(x.shape)
        x = F.log_softmax(x, dim=2)

        return x

    
    def train(self, num_epochs, data_loader, optimizer, loss_fn):
        losses = collections.defaultdict(list)
        for epoch in range(num_epochs):
            iterator = tqdm(data_loader)
            for data, label in iterator:
                data = data.to(DEVICE)
                label = label.to(DEVICE)

                optimizer.zero_grad()
                
                output = self(data)
    
                T = output.size(0)
                N = output.size(1)
            
                input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.int32)
                #target_lengths = torch.full(size=(N,), fill_value=5, dtype=torch.int32)
                target_lengths = torch.tensor([get_length(l) for l in label])
    
                loss = loss_fn(output, label, input_lengths, target_lengths)

                loss.backward()

                # torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

                optimizer.step()
    
                iterator.set_postfix({'Epoch':epoch+1, 'Loss' : loss.item()})
                losses['loss'].append(loss.item())
        
        losses_df = pd.DataFrame(losses)
        losses_df.to_csv('./losses_upgrade.csv')

        