"""
Nathan Rayon
4/20/2025

PyTorch VQA Dataset
"""

import torch
from torchvision.transforms import functional as FT
from torchvision.transforms import v2 as T
from torch.nn import functional as F
from torch.utils.data import Dataset
from PIL import Image
import cv2
import albumentations as A  # our data augmentation library
from albumentations.pytorch import ToTensorV2

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

from collections import defaultdict, deque
from torchvision.utils import draw_bounding_boxes

import json
import re
import glob

with open(r"F:\UAVProject\AirSim\PythonClient\EntropyRewritePlusVQA\Dataset\answers_train_sequential_uav.txt", "r") as text_data:
  # Split text data by lines and remove empty lines
  text_content = text_data.read()

  lines = [line.strip() for line in text_content.splitlines() if line.strip()]

# Create a class dictionary where each unique answer is assigned a unique class ID
CLASS_DICT = {value: idx for idx, value in enumerate(sorted(set(lines)))}

print(CLASS_DICT)


def preprocess_image(image_path):
    im = Image.open(image_path).convert('RGB')
    im = im.resize((512, 512))
    im = F.to_tensor(im)
    im = np.array(im)
    return im

def read_images(path_to_img):
    img = preprocess_image(path_to_img)
    return img

def get_transforms():
    
    transform = A.Compose([
        A.Resize(600, 600), 
        ToTensorV2()
    ])
    return transform

class VQADataset(Dataset):
    def __init__(self, dataset_dir, split, anno_file_name, QandA_file_name):
      self.img_dir = os.path.join(dataset_dir, split)
      self.coco = glob.glob(os.path.join(self.img_dir)) # annotatiosn stored here
    

      self.anno_dir = os.path.join(dataset_dir, split, anno_file_name)
      self.dataset_frame = pd.read_csv(os.path.join(dataset_dir, split, anno_file_name))
      self.questions_and_answers = json.load(open(QandA_file_name))
      self.question_vocab = self._generate_question_vocab()
      self.answer_vocab = self._generate_answer_vocab()
      # print(self.question_vocab)
      # print(self.answer_vocab)

    def __len__(self):
        return len(self.questions_and_answers)
    
    def _generate_question_vocab(self):
      question_vocab = {}
      count = 0
      for text in self.questions_and_answers:
        full_string = text[0]
        full_string = re.sub('\W+', ' ', full_string)

        tokens = re.split(' ', full_string)
        tokens.remove('')

        for token in tokens:
            if token.lower() not in question_vocab:
                question_vocab[token.lower()] = count
                count += 1
      return question_vocab    
    
  

    def _generate_answer_vocab(self):

      with open(r"F:\UAVProject\AirSim\PythonClient\EntropyRewritePlusVQA\Dataset\answers_train_sequential_uav.txt", "r") as text_data:
        # Split text data by lines and remove empty lines
        text_content = text_data.read()

      lines = [line.strip() for line in text_content.splitlines() if line.strip()]

      # Create a class dictionary where each unique answer is assigned a unique class ID
      answer_vocab = {value: idx for idx, value in enumerate(sorted(set(lines)))}
       
      return answer_vocab
      # self.answer_vocab = CLASS_DICT
    

    def _load_image(self, idx):
      # path = self.coco[idx]
      path = os.path.join(self.img_dir, idx)
      # print(path)
      image = cv2.imread(path)
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      transform = A.Compose([
        A.Resize(600, 600),  # Resize image to 600x600
        ToTensorV2()  # Convert image to tensor
      ])
      augmented = transform(image=image)
      image = augmented['image']
      return image.div(255)
    
    def _load_question(self, idx):
      question_vector = np.zeros(len(self.question_vocab))

      full_string = self.questions_and_answers[idx][0]
      # print(full_string)

      full_string = re.sub('\W+', ' ', full_string)

      tokens = re.split(' ', full_string)
      tokens.remove('')
      for word in tokens:
          if word.lower() in self.question_vocab:
              question_vector[self.question_vocab[word.lower()]] += 1
      
      return question_vector


    def _load_answer(self, idx):
      answer_vector = np.zeros(len(self.answer_vocab))

      full_string = self.questions_and_answers[idx][1]
      # print(full_string)

      tokens = [token.strip() for token in full_string.split(",")]

      # print(tokens)
      for token in tokens:
        if token in self.answer_vocab and answer_vector[self.answer_vocab[token]] == 0:
          # print("FOUND")
          answer_vector[self.answer_vocab[token]] += 1

      # print(answer_vector)
      return answer_vector

    def __getitem__(self, idx):
      # print(self.questions_and_answers[idx][2])
      image = self._load_image(self.questions_and_answers[idx][2])
      question_vector = self._load_question(idx)
      answer_vector = self._load_answer(idx)
      # print(question_vector)
      # print(answer_vector)
      return image, torch.tensor(question_vector, dtype=torch.float32), torch.tensor(answer_vector, dtype=torch.float32)
    
    def get_answer_from_vector(self, predicted_answer):
      answers = []
   
      return [key for key, val in self.answer_vocab.items() if val == predicted_answer]

    
    def get_question_vector(self, question):
      question_vector = np.zeros(len(self.question_vocab))

      full_string = question

      full_string = re.sub('\W+', ' ', full_string)

      tokens = re.split(' ', full_string)
      tokens.remove('')
      for word in tokens:
          if word.lower() in self.question_vocab:
              question_vector[self.question_vocab[word.lower()]] += 1
      
      return question_vector