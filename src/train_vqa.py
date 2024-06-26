import torch
import torchvision
import torchvision.transforms as T
from torchvision.transforms import functional as F
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


def train_loop(model, optimizer, criterion, train_loader):
    model.train()
    model.to(device)
    total_loss, total = 0, 0
    count = 0

    for image, text, label in trainloader:
        count += 1




        
        
        # get the inputs; data is a list of [inputs, labels]
        image, text, label =  image.to(device), text.to(device), label.to(device)

        with torch.no_grad():
            features = rcnnmodel.backbone(image)
            features = features['pool'].view(features['pool'].size(0), -1)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = model.forward(features, text)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        # Record metrics
        total_loss += loss.item()
        total += len(label)

    return total_loss / total


def validate_loop(model, criterion, valid_loader):
    model.eval()
    model.to(device)
    total_loss, total = 0, 0

    print("Validating!")


    with torch.no_grad():
      for image, text, label in testloader:
          # get the inputs; data is a list of [inputs, labels]
          image, text, label =  image.to(device), text.to(device), label.to(device)

          with torch.no_grad():
            features = rcnnmodel.backbone(image)
            features = features['pool'].view(features['pool'].size(0), -1)
            print(features.shape)
            print(text.shape)
          # Forward pass
          output = model.forward(features, text)

          # Calculate how wrong the model is
          loss = criterion(output, label)

          # Record metrics
          total_loss += loss.item()
          total += len(label)

    return total_loss / total

from torch.utils.data import DataLoader

from tqdm.notebook import tqdm



if torch.cuda.is_available(): device = torch.device("cuda:0")
kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}


# training and test datasets and data loaders
train_dataset = VQA('..\\Dataset\\VQA_New_Dataset_Aerial.v1i.coco', 'train', 'train_annotations.csv', '..\\Dataset\\train_questions_and_answers.txt')
test_dataset = VQA('..\\Dataset\\VQA_New_Dataset_Aerial.v1i.coco', 'test', 'test_annotations.csv', '..\\Dataset\\test_questions_and_answers.txt')
trainloader = DataLoader(train_dataset, shuffle=True, batch_size=4)
testloader = DataLoader(test_dataset, batch_size=4)


# Initialize model, recursively go over all modules and convert their parameters and buffers to CUDA tensors
model = VQAModel(17, 17, 14).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01,
                      momentum=0.5 )


train_losses, valid_losses = [], []

print("TRAINING START: Number of Epochs")

epoch_range = 5
print(epoch_range)

for epoch in range(int(epoch_range)):
    train_loss = train_loop(model, optimizer, criterion, trainloader)
    valid_loss = validate_loop(model, criterion, testloader)

    tqdm.write(
        f'epoch #{epoch + 1:3d}\ttrain_loss: {train_loss:.2e}\tvalid_loss: {valid_loss:.2e}\n',
    )

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    print(f'Epoch: {epoch},\n')
    print(f'Training Loss: {train_loss},\n')
    print(f'Validation Loss: {valid_loss}\n')


import matplotlib.pyplot as plt
plt.style.use('ggplot')


epoch_ticks = range(1, epoch + 2)
plt.plot(epoch_ticks, train_losses)
plt.plot(epoch_ticks, valid_losses)
plt.legend(['Train Loss', 'Valid Loss'])
plt.title('Losses')
plt.xlabel('Epoch #')
plt.ylabel('Loss')
plt.xticks(epoch_ticks)
plt.show()

model.eval()
model.to(device)
# num_correct = 0
# num_samples = 0
# predictions = []
# answers = []

# with torch.no_grad():
#     for image, text, label in testloader:
#         image, text, label =  image.to(device), text.to(device), label.to(device)
#         probs = model.forward(image, text)

#         _, prediction = probs.max(1)
#         predictions.append(prediction)

#         answer = torch.argmax(label, dim=1)
#         answers.append(answer)

#         num_correct += (prediction == answer).sum()
#         num_samples += prediction.size(0)

#     valid_acc = (f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')
#     print(valid_acc)


torch.save(model.state_dict(), 'model_vqa1.pt')

