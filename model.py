import torch.nn.functional as F
from dataset import *


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3)
        self.conv1_coef = nn.Parameter(torch.randn(64, device=device, dtype=dtype))
        self.conv2_coef = nn.Parameter(torch.ones(192, device=device, dtype=dtype))
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x, flag):
        conv1_coef = self.conv1_coef.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        conv2_coef = self.conv2_coef.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        x = F.max_pool2d(F.relu(self.conv1(x) * conv1_coef), (2, 2)) if flag is True \
            else F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x) * conv2_coef), 2) if flag is True \
            else F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
