import torch
from torch.autograd import Variable


class WeightedL1Loss(torch.nn.Module):
    def __init__(self, weight=100):
        super(WeightedL1Loss, self).__init__()
        self.weight = weight

    def forward(self, output, label):
        mask = (label > 0).float()
        loss = torch.abs(output - label) * mask * self.weight
        return loss.mean()