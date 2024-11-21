import os
import glob 
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T

def predict(image, model, mapping_inv, DEVICE):
        image = Image.open(image).convert('L')
        image_tensor = T.Resize((80, 500))(image)
        image_tensor = T.ToTensor()(image_tensor)
        image_tensor = image_tensor.unsqueeze(0)        
        out = model(image_tensor.to(device=DEVICE))
        out = out.permute(1, 0, 2)
        # out = out.log_softmax(2)
        out = out.argmax(2)
        out = out.cpu().detach().numpy()[0]
        # print(out)

        pred = ''
        then = 0
        for x in out:
            if then != x:
                if x > 0 :
                    pred += mapping_inv[x]
            then = x
        
        return pred 

def get_accuracy(model, mapping_inv, DEVICE):
    data = glob.glob(os.path.join('./test', '*.png'))
    correct = 0
    total_characters = 0
    correct_characters = 0
    iterator = tqdm(data)
    for d in iterator:
        pred = predict(d, model, mapping_inv, DEVICE)
        actual = d.split('/')[-1].split('-')[0]
        correct_pred_count = 0
        for i in range(min(len(pred), len(actual))):
            if pred[i] == actual[i]:
                correct_pred_count += 1
        if correct_pred_count == len(actual):
            correct += 1
        total_characters += len(actual)
        correct_characters += correct_pred_count