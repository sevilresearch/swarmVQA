import torch
import torchvision
from torchvision import datasets, models
from torchvision.transforms import functional as FT
from torchvision.transforms import v2 as T
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, sampler, random_split, Dataset
import copy
import math
from PIL import Image
import cv2
import albumentations as A  # our data augmentation library
from albumentations.pytorch import ToTensorV2


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
import json
import re
import glob


CLASS_DICT = {
   'SUV': 0,           
   'Cargo Truck': 1,
   'Sports Car': 2,
   'Pickup': 3
}

def preprocess_image(image_path):
    im = Image.open(image_path).convert('RGB')
    im = im.resize((512, 512))
    im = F.to_tensor(im)
    im = np.array(im)
    return im

def read_images(path_to_img):
    img = preprocess_image(path_to_img)
    return img

def get_transform(train):
    transforms = []
    # if train:
    #     transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=False))
    # transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

class VQA(Dataset):
    def __init__(self, dataset_dir, split, anno_file_name, QandA_file_name):
      self.img_dir = os.path.join(dataset_dir, split)
      self.coco = glob.glob(os.path.join(self.img_dir, "*.jpg")) # annotatiosn stored here
      
      self.anno_dir = os.path.join(dataset_dir, split, anno_file_name)
      self.dataset_frame = pd.read_csv(os.path.join(dataset_dir, split, anno_file_name))
      self.questions_and_answers = json.load(open(QandA_file_name))
      self.question_vocab = self._generate_question_vocab()
      self.answer_vocab = self._generate_answer_vocab()

    def __len__(self):
        return len(self.questions_and_answers)
    
    def _generate_question_vocab(self):
      questions_and_answers = json.load(open(r"C:\Users\Nathon Rayon\.conda\envs\VQATorch\ThesisVQA\Dataset\train_questions_and_answers.txt"))
      question_vocab = {}
      count = 0
      for text in questions_and_answers:
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
      questions_and_answers = json.load(open(r"C:\Users\Nathon Rayon\.conda\envs\VQATorch\ThesisVQA\Dataset\train_questions_and_answers.txt"))

      answer_vocab = {}
      count = 0
      for text in questions_and_answers:
        full_string = text[1]
        full_string = re.sub('\W+', ' ', full_string)

      
        if full_string.lower() not in answer_vocab:
            answer_vocab[full_string.lower()] = count
            count += 1
      return answer_vocab   
    
    def _load_image(self, idx):

      path = self.coco[idx]
      image = cv2.imread(path)
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      image = T.ToTensor()(image)
      return image.div(255)
    
    def _load_question(self, idx):
      question_vector = np.zeros(len(self.question_vocab))

      full_string = self.questions_and_answers[idx][0]

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

      full_string = re.sub('\W+', ' ', full_string)
    
      if full_string.lower() in self.answer_vocab:
          answer_vector[self.answer_vocab[full_string.lower()]] += 1
        
      return answer_vector

    def __getitem__(self, idx):
      #Index divided by 7 since we are asking 7 questions per image

      image = self._load_image(self.questions_and_answers[idx][2])
      question_vector = self._load_question(idx)
      answer_vector = self._load_answer(idx)
   
      return image, torch.tensor(question_vector, dtype=torch.float32), torch.tensor(answer_vector, dtype=torch.float32)
    
    def get_answer_from_vector(self, idx):
       return [key for key, val in self.answer_vocab.items() if val == idx]
    
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

# vqaDataset = VQA('VQA_New_Dataset_Aerial.v1i.coco', 'train', 'train_annotations.csv', 'train_questions_and_answers.txt')
# print(vqaDataset[0])
# questionDF = pd.read_csv(os.path.join('VQA_New_Dataset_Aerial.v1i.coco', 'test', 'test_annotations.csv'))

# # questionDF = vqaDataset[0]
# print(questionDF.head())
# # questionDF = pd.DataFrame(questionDF['filename'].unique(), columns=['filename'])
# # questionDF.columns = ['filename']
# # print(len(questionDF['filename']))

# grouped_df = questionDF.groupby('filename').agg({
#     'class': lambda x: list(x), 
# }).reset_index()

# # print(grouped_df.head(50))

# def create_questions(Class, Image_ID):
#     #Creating types of questions for model
#     questions = [
#         (f'What vehicles are in this image?', ', '.join(list(set(Class)))),
#         (f'How many cars are there?', str(len(Class))),
#         (f'Is there a red car?', 'yes' if 'SportsCar' in Class else 'no'),
#         (f'Is there a green car?', 'yes' if 'SUV' in Class else 'no'),
#         (f'Is there a white car?', 'yes' if 'CargoTruck' in Class else 'no'),
#         (f'Is there a yellow car?', 'yes' if 'Pickup' in Class else 'no'),
#     ] 

#     return (list(map(lambda x: x + (Image_ID,), questions)))



# all_questions = []

# for index, row in grouped_df.iterrows():
#    all_questions += create_questions(row['class'], index)

# # print(all_questions)


# question_vocab = {}
# count = 0
# for text in all_questions:
#    full_string = text[0]
#    full_string = re.sub('\W+', ' ', full_string)
# #    print(full_string)
#    tokens = re.split(' ', full_string)
#    tokens.remove('')
# #    print(tokens)
#    for token in tokens:
#       if token.lower() not in question_vocab:
#          question_vocab[token.lower()] = count
#          count += 1
# # print(question_vocab)



# all_answers = list(set(map(lambda q: q[1], all_questions)))


# with open('test_questions_and_answers.txt', 'w') as file:
#     json.dump(all_questions, file)

# # with open('questions_and_answers_test.txt', 'w') as file:
# #     json.dump(all_questions_test, file)

# with open('test_answers.txt', 'w') as file:
#   for answer in all_answers:
#     file.write(f'{answer}\n')