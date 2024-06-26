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

coco = COCO(os.path.join(dataset_path, "train", "_annotations.coco.json"))
categories = coco.cats
n_classes = len(categories.keys())
print(categories)

classes = [i[1]['name'] for i in categories.items()]
print(classes)