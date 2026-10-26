import torch
import torch.nn as nn
import torchvision.models as models

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channel, channel // reduction)
        self.fc2 = nn.Linear(channel // reduction, channel)
    
    def forward(self, x):
        b, c, _, _ = x.shape
        y = x.mean([2, 3])
        y = torch.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y

class PneumoniaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.resnet50(weights='IMAGENET1K_V1')
        
        # Insert SE attention after last residual block
        self.se = SEBlock(2048)
        
        # Custom head with dropout for regularization
        self.base.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 1)
        )
    
    def forward(self, x):
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)
        
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        
        x = self.se(x)          # Attention module
        
        x = self.base.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.base.fc(x)
        return x