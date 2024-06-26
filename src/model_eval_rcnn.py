import torch
import torchvision
from torchvision import datasets, models
from torchvision.transforms import functional as FT
from torchvision import transforms as T
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, sampler, random_split, Dataset
import copy
import math
from PIL import Image
import cv2
import albumentations as A  # our data augmentation library
from albumentations.pytorch import ToTensorV2
import sys

sys.path.insert(1, r'C:\Users\Nathon Rayon\.conda\envs\VQATorch\ThesisVQA\Dataset')

from vision_dataset import VisionDataset

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

from collections import defaultdict, deque
import datetime
import time
from tqdm import tqdm # progress bar
from torchvision.utils import draw_bounding_boxes

from pycocotools.coco import COCO

import matplotlib.pyplot as plt

dataset_path = r'C:\Users\Nathon Rayon\.conda\envs\VQATorch\ThesisVQA\Dataset\VQA_New_Dataset_Aerial.v1i.coco'

coco = COCO(os.path.join(dataset_path, "test", "_annotations.coco.json"))
categories = coco.cats

n_classes = len(categories.keys())
print(categories)

classes = [i[1]['name'] for i in categories.items()]


model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features # we need to change the head
model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, n_classes)

model.load_state_dict(torch.load(r'C:\Users\Nathon Rayon\.conda\envs\VQATorch\VQA\model\model_new_v3.pt'))
model.eval()
torch.cuda.empty_cache()


test_dataset = RCNNDataset(root=dataset_path, split="test", transforms=get_transforms(False))

device = torch.device("cuda") # use GPU to train
model.to(device)

img, _ = test_dataset[4]
img_int = torch.tensor(img*255, dtype=torch.uint8)
with torch.no_grad():
    prediction = model([img.to(device)])
    features = model.backbone(img.to(device))
    print((features['0'].shape))
    print((features['1'].shape))
    # print((features['pool'][0]))
    # print(features['pool'].view(features['pool'].size(0), -1).shape)
    # fc_visual = nn.Linear(256*19*19, 512)
    # reducedFeatures = torch.relu(fc_visual(features['pool'].view(features['pool'].size(0), -1)))
    # print(reducedFeatures)

    # print(prediction)
    pred = prediction[0]

print(features['pool'].view(features['pool'].size(0), -1).shape)
fc_visual = nn.Linear(25600, 16384).to(device)
reducedFeatures = torch.relu(fc_visual(features['pool'].view(features['pool'].size(0), -1)))
print(reducedFeatures.shape)

# print(pred['features'])
fig = plt.figure(figsize=(14, 10))
plt.imshow(draw_bounding_boxes(img_int,
    pred['boxes'][pred['scores'] > 0.5],
    [classes[i] for i in pred['labels'][pred['scores'] > 0.5].tolist()], width=4
).permute(1, 2, 0))
plt.show()


train_anno = pd.read_csv(r"..\Dataset\VQA_New_Dataset_Aerial.v1i.coco\train_annotations.csv")
print(train_anno.head())

#Create questions BoW Vector
#Example:
# "I saw a tree", "I saw a bird"
# BoW Vector = [0, 0, 0, 0, 0] Length of 5 for vocab, each index is a word in vocabulary
# Answer Vector is 0 or 1 depending on which is the right answer
# Need to create vector in dataset.

BoWVec = []
