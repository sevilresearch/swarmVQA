
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn


class VQAModel(nn.Module):
    def __init__(self, text_dim, hidden_dim, num_answers):
        super(VQAModel, self).__init__()
        self.fc1_visual = nn.Linear(25600, 12800)
        self.fc2_visual = nn.Linear(12800, 6400)
        self.fc3_visual = nn.Linear(6400, 3200)
        self.fc4_visual = nn.Linear(3200, 1600)
        self.fc5_visual = nn.Linear(1600, 800)
        self.fc6_visual = nn.Linear(800, 400)
        self.fc7_visual = nn.Linear(400, 200)
        self.fc8_visual = nn.Linear(200, 100)
        self.fc9_visual = nn.Linear(100, 50)
        self.fc10_visual = nn.Linear(50, 25)
        self.fc11_visual = nn.Linear(25, 17)
        
        self.fc_text = nn.Linear(text_dim, text_dim)
        self.fc_combined = nn.Linear(hidden_dim * 2, num_answers)

    
    def forward(self, visual_features, text_features):
        visual_features1 = F.relu(self.fc1_visual(visual_features))
        visual_features2 = F.relu(self.fc2_visual(visual_features1))
        visual_features3 = F.relu(self.fc3_visual(visual_features2))
        visual_features4 = F.relu(self.fc4_visual(visual_features3))
        visual_features5 = F.relu(self.fc5_visual(visual_features4))
        visual_features6 = F.relu(self.fc6_visual(visual_features5))
        visual_features7 = F.relu(self.fc7_visual(visual_features6))
        visual_features8 = F.relu(self.fc8_visual(visual_features7))
        visual_features9 = F.relu(self.fc9_visual(visual_features8))
        visual_features10 = F.relu(self.fc10_visual(visual_features9))
        visual_features11 = F.relu(self.fc11_visual(visual_features10))

        text_features = F.relu(self.fc_text(text_features))

        combined_features = torch.cat((visual_features11, text_features), dim=1)

        output = self.fc_combined(combined_features)
        return output


