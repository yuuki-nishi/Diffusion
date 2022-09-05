import torch.nn as nn
import torch.nn.functional as F
import makeconfig

class Classifier(nn.Module):    
    def __init__(self,config : makeconfig.Myconfig):
        super().__init__()
        self.fc1 = nn.Linear(4, 50)
        self.fc2 = nn.Linear(50, 3)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x