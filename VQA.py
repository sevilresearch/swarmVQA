"""
Nathan Rayon
4/20/2025

VQA API
"""

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

from Dataset.VQADataset import VQADataset

class VQA:

    def __init__(self, cv_model, vqa_model):

        self.cv_model = cv_model
        self.vqa_model = vqa_model
        self.vqa_dataset = VQADataset('EntropyRewritePlusVQA/Dataset/UAV_Dataset', 'test', 'test_annotations.csv', 'EntropyRewritePlusVQA/Dataset/questions_and_answers_test_sequential_uav.txt')
        self.device = torch.device("cuda")

        self.dataset_path = r'C:\Users\NATHANRAYON\miniconda3\envs\ThesisEnv\ThesisVQA\ThesisVQATemp\swarmVQA\Dataset\UAV_Dataset'

        coco = COCO(os.path.join(self.dataset_path, "test", "_annotations.coco.json"))
        categories = coco.cats

        n_classes = len(categories.keys())

        self.classes = [i[1]['name'] for i in categories.items()]


        self.vqa_model.to(self.device)
        self.cv_model.to(self.device)
        
        self.prev_images = []
        self.questions = []
        self.current_round = 0
        self.prev_answers = []

        self.current_image = None
        self.current_answers = []

        self.experiment_answers = [
                                    "Old Woman",
                                    "No people.",
                                    "Young Man",
                                    "0",
                                    "SUV",
                                    "2",
                                    "Old Woman",
                                    "Hatchback",
                                    "no",
                                    "Passenger",
                                    "No cars.",
                                    "yes",
                                    "Young Woman",
                                    "Cargo Truck",
                                    "Commercial",
                                    "Blue",
                                    "Old Man",
                                    "White",
                                    "Green",
                                    "1"
                                   ]

        self.model_answers = set()
   
    #Use any accuracy gathering statistic you want here.
    def get_vqa_accuracy(self):
        
        #Simple accuracy getter.
        accuracy = (len(self.model_answers) / 21) * 100

        return accuracy

    def vectorize_cv_output(self, pred_classes):
        cv_vector = torch.zeros(len(self.classes), dtype=torch.float32)
        for i, cv_class in enumerate(self.classes):
            if cv_class in pred_classes:
                cv_vector[i] = 1
        
        return cv_vector
    

    def predict(self):

        questions = [
        f'How many people are in this image?',
        f'How many cars are in this image?',
        f'Are there any people in this image?',
        f'Are there any women in this image?',
        f'Are there any men in this image?',
        f'What color is the car in this image?',
        f'What is the age group of the person in this image?',
        f'Is the car a commercial or passenger vehicle?',
        f'What cars are in this image?'
        ]
       
        for question in questions:
            question = torch.tensor(self.vqa_dataset.get_question_vector(question), dtype=torch.float32)

       
            with torch.no_grad():
                cv_features = []

                img = self.current_image.to(self.device)
        
                prediction = self.cv_model([img])
                pred = prediction[0]

                pred_classes = [self.classes[i] for i in pred['labels'][pred['scores'] > 0.6].tolist()]

                cv_vector = self.vectorize_cv_output(pred_classes)
                cv_features.append(cv_vector)
                cv_features = torch.stack(cv_features)
                cv_features = cv_features.to(self.device)

                pred = self.vqa_model(cv_features.to(self.device), question.to(self.device).unsqueeze(0))
                # Apply softmax to get probabilities
                probabilities = torch.sigmoid(pred)


                # Determine the predicted answer (class with the highest probability)
                predicted_classes = (probabilities > 0.6).float()

                indices = torch.nonzero(predicted_classes[0], as_tuple=True)[0]  # Check the first (and only) element in the batch

                predicted_answers = [self.vqa_dataset.get_answer_from_vector(idx.item()) for idx in indices]
                flat_answers = [item for sublist in predicted_answers for item in sublist]

                for answer in flat_answers:
                    if answer in self.experiment_answers:
                        print("Answer found: ", answer)
                        print(f"Answer set size: {len(self.model_answers)}")
                        self.model_answers.add(answer)


                # Print all predicted answers, joined by commas
                print(", ".join(flat_answers))

    def transform(self, img):
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        transform = A.Compose([
        A.Resize(600, 600),  # Resize image to 600x600
        ToTensorV2()  # Convert image to tensor
        ])
        augmented = transform(image=image)
        image = augmented['image']
        return image.div(255)

    #Adds images to the current_images array before processing them.       
    def add_image(self, image):
        
        self.current_image = image


    def add_answers(self, answers):
        self.answers.append(answers)
