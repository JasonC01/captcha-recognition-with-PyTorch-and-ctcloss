import os
import numpy as np
from PIL import Image
import torch

class CaptchaDataset:
    def __init__(self, df, max_len, path, transform=None):
        self.df = df
        self.transform = transform
        self.max_len = max_len
        self.path = path
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        image = Image.open(os.path.join(self.path, data['image'])).convert('L')
        label = torch.tensor(data['label'], dtype=torch.int32)
        label = torch.cat((label, torch.tensor([0] * (self.max_len - len(label)))))
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) 
        # sharpened_image = Image.fromarray(cv2.filter2D(np.array(image), -1, kernel))
        if self.transform is not None:
            image = self.transform(image)
            # sharpened_image = self.transform(sharpened_image)
                
        return image, label
        