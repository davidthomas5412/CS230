import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, padding=0):
        super(CrossEntropyLoss2d, self).__init__()
        self.padding = padding
        self.nll_loss = nn.NLLLoss2d(None, True)

    def forward(self, inputs, targets):
        if self.padding > 0:
            return self.nll_loss(F.log_softmax(inputs[:, :, self.padding:-self.padding, self.padding:-self.padding]), targets)
        return self.nll_loss(F.log_softmax(inputs), targets)


if __name__ == '__main__':
    import torch
    from math import log, exp

    criterion = CrossEntropyLoss2d(size_average=True)

    inp = Variable(torch.FloatTensor([[[[0.5]], [[0.1]]], [[[0.7]], [[0.1]]]]))
    out = Variable(torch.LongTensor([[[0]],[[1]]]))

    c = CrossEntropyLoss2d()
    print(c(inp, out))
    print(-(log(exp(0.5) / (exp(0.5) + exp(0.1))) + log(exp(0.1) / (exp(0.7) + exp(0.1))))/ 2)