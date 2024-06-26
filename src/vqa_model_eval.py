import torch
import torchvision
import torchvision.transforms as T
from torchvision.transforms import functional as F
import torch.nn.functional as F2
from PIL import Image
import os
from os import path
import pandas as pd
from random import randint
import json
import torchvision.utils as utils
import numpy as np
import sys
import albumentations as A 
from albumentations.pytorch import ToTensorV2
from pycocotools.coco import COCO
from torchvision import datasets, models

import cv2

sys.path.insert(1, r'C:\Users\Nathon Rayon\.conda\envs\VQATorch\ThesisVQA\Dataset')
sys.path.insert(2, r'C:\Users\Nathon Rayon\.conda\envs\VQATorch\ThesisVQA\Model')
sys.path.insert(3, r'C:\Users\Nathon Rayon\.conda\envs\VQATorch\ThesisVQA\utils')

from utility import get_transforms

from vision_dataset import VisionDataset
from VQAModel import VQAModel
from VQADataset import VQA


from sentence_transformers import SentenceTransformer, util

device = torch.device("cuda")

if torch.cuda.is_available(): 
    device = torch.device("cuda:0")
    print(torch.cuda.get_device_name(device))





def load_image():


    path = r'C:\Users\Nathon Rayon\.conda\envs\VQATorch\ThesisVQA\Dataset\VQA_New_Dataset_Aerial.v1i.coco\test\Screenshot-2024-04-24-135555_png.rf.4534a803bc6cdf5ebd3affdf9d6d61a1.jpg'

    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = T.ToTensor()(image)
    
    return image.div(255)

dataset_path = r'C:\Users\Nathon Rayon\.conda\envs\VQATorch\ThesisVQA\Dataset\VQA_New_Dataset_Aerial.v1i.coco'

coco = COCO(os.path.join(dataset_path, "test", "_annotations.coco.json"))
categories = coco.cats


n_classes = len(categories.keys())
print(categories)

classes = [i[1]['name'] for i in categories.items()]


rcnnmodel = models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
in_features = rcnnmodel.roi_heads.box_predictor.cls_score.in_features # we need to change the head
rcnnmodel.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, n_classes)

rcnnmodel.load_state_dict(torch.load(r'C:\Users\Nathon Rayon\.conda\envs\VQATorch\VQA\model\model_new_v3.pt'))
rcnnmodel.eval()
torch.cuda.empty_cache()

device = torch.device("cuda") # use GPU to train
rcnnmodel.to(device)

from torch.utils.data import DataLoader

from tqdm.notebook import tqdm



if torch.cuda.is_available(): device = torch.device("cuda:0")
kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}


# training and test datasets and data loaders
test_dataset = VQA('..\\Dataset\\VQA_New_Dataset_Aerial.v1i.coco', 'test', 'test_annotations.csv', '..\\Dataset\\test_questions_and_answers.txt')

device = torch.device("cuda") # use GPU to train

# Initialize model, recursively go over all modules and convert their parameters and buffers to CUDA tensors
model = VQAModel(17, 17, 14).to(device)
model.load_state_dict(torch.load(r'C:\Users\Nathon Rayon\.conda\envs\VQATorch\ThesisVQA\src\model_vqa.pt'))
model.to(device)

img = load_image()
question = torch.tensor(test_dataset.get_question_vector("How many red cars are there?"), dtype=torch.float32)



print(question.to(device))
with torch.no_grad():
    features = rcnnmodel.backbone(img.to(device))
    features = features['pool'].view(features['pool'].size(0), -1)
    print(features.shape)
    pred = model(features.to(device), question.to(device).unsqueeze(0))
    # Apply softmax to get probabilities
    probabilities = F2.softmax(pred, dim=1)

    # Determine the predicted answer (class with the highest probability)
    predicted_class = torch.argmax(probabilities, dim=1)

    print("Probabilities:", probabilities)
    print("Predicted class:", predicted_class.item())
    print(test_dataset.get_answer_from_vector(predicted_class.item()))
train_losses, valid_losses = [], []

print("TRAINING START: Number of Epochs")

epoch_range = 5
print(epoch_range)
