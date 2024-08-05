import torch
from torch import nn


class SLNN(nn.Module):
    def __init__(self, input_size):
        super(SLNN, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x)).squeeze(-1)


class MLNN(nn.Module):
    def __init__(self, input_size, hidden_size=2):
        super(MLNN, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        hidden = self.sigmoid(self.hidden(x))
        return self.sigmoid(self.output(hidden)).squeeze(-1)


def get_pre_activation_slnn(model, x, model_name="slnn"):
    assert model_name == "slnn"
    with torch.no_grad():
        pre_activation = model.linear(x)
    return pre_activation.squeeze(-1)


def get_pre_activation_mlnn(model, x):
    pre1 = model.sigmoid(model.hidden(x))
    pre3 = model.output(pre1)
    return pre1, pre3
