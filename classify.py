import torch
import torch.nn as nn 
from torchsummary import summary

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(2,64)
        self.fc2 = nn.Linear(64,2)
        self.relu = nn.ReLU()
    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
if __name__ == '__main__':
    model = Net()
    summary(model,(600,2))