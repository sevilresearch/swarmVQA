"""
Nathan Rayon
4/20/2025

PyTorch VQA model architecture
"""



import torch.nn as nn
import torch.nn.functional as F
import torch


class VQAModel(nn.Module):
    def __init__(self, text_dim, hidden_dim, num_answers):
        super(VQAModel, self).__init__()

        self.fc1_visual = nn.Linear(9, 16)
        self.fc2_visual = nn.Linear(16, 32)
          
        self.fc_text = nn.Linear(text_dim, 32)
        self.fc_combined = nn.Linear(hidden_dim * 2, num_answers)

    
    def forward(self, visual_features, text_features):

        visual_features1 = F.relu(self.fc1_visual(visual_features))
        visual_features2 = F.relu(self.fc2_visual(visual_features1))
    
        text_features = F.relu(self.fc_text(text_features))

        combined_features = torch.cat((visual_features2, text_features), dim=1)

        output = self.fc_combined(combined_features)


        return output