import numpy as np
import pandas as pd
import torch
from torchnet.meter.meter import Meter
import sklearn.metrics as metrics


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
        return self.running_loss / self.num_samples


class SklearnMeter:
    def __init__(
        self, func, tensor=None, proba2int=True, reduction='mean'
    ):
        super(SklearnMeter, self).__init__()
        self.proba2int = proba2int
        self.func = func
        self.tensor = tensor
        self.reset()

    def __call__(self, output, target):
        self.reset()
        self.add(output, target)
        res = self.value()
        self.reset()
        return res

    def reset(self):
        self.outputs = []
        self.targets = []
        self.ids = []

    def add(self, output, target):
        if self.tensor is None:
            self.tensor = isinstance(output, torch.Tensor)
        if self.tensor:
            output = output.detach().cpu().numpy()
            target = target.detach().cpu().numpy()
        self.outputs.append(output)
        self.targets.append(target)

    def value(self):
        outputs = np.concatenate(self.outputs, axis=0)
        targets = np.concatenate(self.targets, axis=0)
        if self.proba2int:
            outputs = outputs.argmax(axis=1)
        return self.func(targets, outputs)


class Accuracy(SklearnMeter):
    def __init__(self, proba2int=True, tensor=None):
        def func(y_true, y_pred):
            return metrics.accuracy_score(y_true, y_pred)
        super(Accuracy, self).__init__(func, tensor, proba2int)


class BalancedAccuracy(SklearnMeter):
    def __init__(self, proba2int=True, tensor=None):
        def func(y_true, y_pred):
            return metrics.balanced_accuracy_score(y_true, y_pred)
        super(BalancedAccuracy, self).__init__(func, tensor, proba2int)


class MeanDistance(Meter):
    def __init__(self):
        super(MeanDistance, self).__init__()
        self.tensor = None
        self.reset()

    def __call__(self, output):
        self.reset()
        self.add(output)
        res = self.value()
        self.reset()
        return res

    def reset(self):
        self.data = []

    def add(self, output):
        if self.tensor is None:
            self.tensor = isinstance(output, torch.Tensor)
        if self.tensor:
            output = output.detach().cpu().numpy()
        self.data.append(output)

    def value(self):
        all_data = np.concatenate(self.data, axis=0)
        return self.func(all_data)

    def func(self, all_data):
        diffs = all_data[:, None, :] - all_data[None, :, :]
        distances = (diffs ** 2).sum(-1)
        distances = np.sqrt(distances)
        return distances.mean()


def test():
    a = np.random.rand(100, 1000)
    m = MeanDistance()
    print(m(a))


if __name__ == '__main__':
    test()
