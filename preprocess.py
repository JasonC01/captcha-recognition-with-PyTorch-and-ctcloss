import os
import glob
import shutil
import pandas as pd
import collections
import numpy as np
import cv2

def is_valid(image_path):
  image = apply_edge_detection(image_path)
  label = image_path.split("/")[-1].split("-")[0]
  return not detect_patch(trim_white(image), len(label))

def apply_edge_detection(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(image, threshold1=30, threshold2=100)
    return edges

def detect_patch(image, label_length):
  block_size = 80
  threshold_density = 0.21
  output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
  y_start = 0
  y_end = 80
  jump_size = 20
  for x in range(0, output.shape[1], jump_size):
      block = output[y_start:y_end, x:x+block_size]
      edge_density = np.sum(block > 0) / (block_size * block_size)
      if edge_density > threshold_density:
          return True
  return False



def trim_white(image):
  h = image.shape[0]
  w = image.shape[1]
  y_start = 40
  y_end = 60
  block_length = 30
  output = image
  trim_start = 0
  trim_end = w / block_length
  for x in range(0, w, block_length):
      block = output[y_start:y_end, x:x+block_length]
      block_density = np.sum(block > 0)
      if block_density != 0:
          trim_start = x
          break

  for x in range(w, 0, -block_length):
      block = output[y_start:y_end, x:x+block_length]
      block_density = np.sum(block > 0)
      if block_density != 0:
          trim_end = x
          break
  output = output[:, trim_start:trim_end]
  return output

# Filter valid train data
data = glob.glob(os.path.join('./train', '*.png'))

valid_datas = collections.defaultdict(list)
valid_datas_list = []
for d in data:
  if is_valid(d):
    valid_datas["path"].append(d)
    valid_datas_list.append(d)
valid_dataframe = pd.DataFrame(valid_datas)
valid_dataframe.to_csv('./valid_train_dataset.csv')

for d in data:
    if d.split("/")[-1] in valid_dataframe.values:
        shutil.copyfile(d, "./valid_train_captcha/"+d.split("/")[-1])
    else:
        shutil.copyfile(d, "./invalid_train_captcha/"+d.split("/")[-1])

# Filter valid test data
data = glob.glob(os.path.join('./test', '*.png'))

valid_datas = collections.defaultdict(list)
valid_datas_list = []
for d in data:
  if is_valid(d):
    valid_datas["path"].append(d)
    valid_datas_list.append(d)
valid_dataframe = pd.DataFrame(valid_datas)
valid_dataframe.to_csv('./valid_test_dataset.csv')

for d in data:
    if d.split("/")[-1] in valid_dataframe.values:
        shutil.copyfile(d, "./valid_test_captcha/"+d.split("/")[-1])
    else:
        shutil.copyfile(d, "./invalid_test_captcha/"+d.split("/")[-1])