import numpy as np
import pandas as pd
import torch
from torchnet.meter.meter import Meter
import sklearn.metrics as metrics
import scipy.spatial.distance as dist


class Loss(Meter):
    def __init__(self):
        super(Loss, self).__init__()
        self.tensor = None
        self.reset()

    def reset(self):
        self.running_loss = 0.
        self.num_samples = 0

    def add(self, batch_loss, batch_size):
        if self.tensor is None:
            self.tensor = isinstance(batch_loss, torch.Tensor)
        if self.tensor:
            batch_loss = batch_loss.detach().cpu().numpy()
        self.running_loss += batch_loss * batch_size
        self.num_samples += batch_size

    def value(self):
        if self.num_samples == 0:
            return np.nan
        return self.running_loss / self.num_samples


def mean_distance(mat):
    distance = dist.pdist(mat)
    return np.mean(distance)


def test():
    a = np.random.rand(100, 2)
    print(mean_distance(a))


if __name__ == '__main__':
    test()
