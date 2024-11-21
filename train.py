import os
import glob 
import pandas as pd
import string
import collections
from dataset import CaptchaDataset
import get_accuracy
from model import CRNN
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
import torch.optim as optim


data = glob.glob(os.path.join('../input/valid-captcha-v2/valid_captcha_v2', '*.png'))
path = '../input/valid-captcha-v2/valid_captcha_v2'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

all_letters = string.ascii_lowercase + string.digits

mapping = {}
mapping_inv = {}

images = []
labels = []
datas = collections.defaultdict(list)
max_len = 0

i = 1
for x in all_letters:
    mapping[x] = i
    mapping_inv[i] = x
    i += 1


for d in data:
    x = d.split('/')[-1]
    datas['image'].append(x)
    label = [mapping[i] for i in x.split('-')[0]]
    max_len = max(max_len, len(label))
    datas['label'].append(label)
df = pd.DataFrame(datas)
num_class = len(mapping)

        
transform = T.Compose([
    T.Resize((80, 650)),
    T.ToTensor()
])

train_data = CaptchaDataset(df, max_len=max_len, path=path, transform=transform)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)


model = CRNN(in_channels=1, num_classes=num_class+1).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.CTCLoss()

model.train(20, train_loader, optimizer, loss_fn)

get_accuracy(model, mapping_inv, DEVICE)