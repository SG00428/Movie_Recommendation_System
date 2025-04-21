import torch
import torch.nn as nn

class FusionModel(nn.Module):
    def __init__(self):
        super(FusionModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, mf_score, ncf_score):
        x = torch.stack([mf_score, ncf_score], dim=1).float()
        return self.fc(x).squeeze()