import torch
import torch.nn as nn

class NonLinClassifier(nn.Module):
    def __init__(self, d_in, n_class, d_hidd=16, activation=nn.ReLU(), dropout=0.1, norm='batch'):
        """
        norm : str : 'batch' 'layer' or None
        """
        super(NonLinClassifier, self).__init__()

        self.dense1 = nn.Linear(d_in, d_hidd)

        if norm == 'batch':
            self.norm = nn.BatchNorm1d(d_hidd)
        elif norm == 'layer':
            self.norm = nn.LayerNorm(d_hidd)
        else:
            self.norm = None

        self.act = activation
        self.dropout = nn.Dropout(dropout)
        self.dense2 = nn.Linear(d_hidd, n_class)

        # No Softmax here (previously had one) -- CrossEntropyLoss expects raw
        # logits and applies its own numerically-stable log_softmax internally;
        # softmaxing before it double-squashed gradients. Matches
        # Core-Clustering's core_clustering/models.py::NonLinClassifier exactly
        # (same architecture otherwise, no new/removed learnable parameters,
        # so state_dicts from either trainer load into either module
        # unchanged). Call predict_proba() below wherever an actual
        # probability distribution is needed.
        self.layers = [self.dense1, self.norm, self.act, self.dropout, self.dense2]
        self.net = nn.Sequential(*[x for x in self.layers if x is not None])

    def forward(self, x):
        out = self.net(x)
        return out


def predict_proba(logits):
    return torch.softmax(logits, dim=1)


class LinClassifier(nn.Module):
    def __init__(self, d_in, n_class):
        super(LinClassifier, self).__init__()
        self.dense = nn.Linear(d_in, n_class)

    def forward(self, x):
        out = self.dense(x)
        return out