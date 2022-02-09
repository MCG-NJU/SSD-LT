import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class DotProduct_Classifier(nn.Module):
    
    def __init__(self, num_classes=1000, feat_dim=2048, *args):
        super(DotProduct_Classifier, self).__init__()
        
        self.fc = nn.Linear(feat_dim, num_classes)
        self.scales = Parameter(torch.ones(num_classes))
        for param in self.fc.parameters():
            param.requires_grad = False
        
    def forward(self, x, *args):
        x = self.fc(x)
        x *= self.scales
        return x

def tau_norm_classifier(**kwargs):
    classifier = DotProduct_Classifier(**kwargs)
    return classifier