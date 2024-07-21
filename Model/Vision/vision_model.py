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

sys.path.insert(1, r'C:\Users\Nathon Rayon\.conda\envs\VQATorch\ThesisVQATemp\swarmVQA\Dataset')
sys.path.insert(2, r'C:\Users\Nathon Rayon\.conda\envs\VQATorch\ThesisVQATemp\swarmVQA\utils')

from utility import get_transforms
import constants
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

dataset_path = r'C:\Users\Nathon Rayon\.conda\envs\VQATorch\ThesisVQATemp\SwarmVQA\Dataset\VQA_New_Dataset_Aerial.v1i.coco'

coco = COCO(os.path.join(dataset_path, "train", "_annotations.coco.json"))
categories = coco.cats
n_classes = len(categories.keys())
print(categories)

classes = [i[1]['name'] for i in categories.items()]
print(classes)

train_dataset = VisionDataset(root=dataset_path, split="train", transforms=get_transforms(True))



print(len(train_dataset))

model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features # we need to change the head
model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, n_classes)

def collate_fn(batch):
    return tuple(zip(*batch))

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

device = torch.device("cuda") # use GPU to train

model = model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9, nesterov=True, weight_decay=1e-4)

def train_one_epoch(model, optimizer, loader, device, epoch):
    model.to(device)
    model.train()
    
    all_losses = []
    all_losses_dict = []
    
    for images, targets in tqdm(loader):
        images = list(image.to(device) for image in images)
        targets = [{k: torch.tensor(v).to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets) # the model computes the loss automatically if we pass in targets
        losses = sum(loss for loss in loss_dict.values())
        loss_dict_append = {k: v.item() for k, v in loss_dict.items()}
        loss_value = losses.item()
        
        all_losses.append(loss_value)
        all_losses_dict.append(loss_dict_append)
        
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping trainig") # train if loss becomes infinity
            print(loss_dict)
            sys.exit(1)
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        

        
    all_losses_dict = pd.DataFrame(all_losses_dict) # for printing
    print("Epoch {}, lr: {:.6f}, loss: {:.6f}, loss_classifier: {:.6f}, loss_box: {:.6f}, loss_rpn_box: {:.6f}, loss_object: {:.6f}".format(
        epoch, optimizer.param_groups[0]['lr'], np.mean(all_losses),
        all_losses_dict['loss_classifier'].mean(),
        all_losses_dict['loss_box_reg'].mean(),
        all_losses_dict['loss_rpn_box_reg'].mean(),
        all_losses_dict['loss_objectness'].mean()
    ))


num_epochs=40

for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, train_loader, device, epoch)

torch.save(model.state_dict(), r'C:\Users\Nathon Rayon\.conda\envs\VQATorch\VQA\model\model_new_v10.pt')
