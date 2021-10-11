import numpy as np
import torch

from sklearn.metrics import precision_score


class AvgrageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.avg, self.sum, self.cnt = 0.0, 0.0, 0

    def update(self, val: float, n: int = 1) -> None:
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def top3_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    batch_size = labels.size(0)
    _, preds = logits.topk(3, 1, True, True)
    preds = preds.t()
    correct = preds.eq(labels.view(1, -1).expand_as(preds))

    n_corrects = correct[:3].float().sum()
    return n_corrects.mul_(1.0 / batch_size).item()


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = logits.data.max(1, keepdim=True)[1]
    acc = pred.eq(labels.data.view_as(pred)).cpu().sum()
    return acc / labels.size(0)


def precision(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    return precision_score(labels.data.cpu().detach().numpy(),
                           np.argmax(logits.data.cpu().detach().numpy(), axis=1),
                           labels=np.arange(0, num_classes).tolist(),
                           average='macro', zero_division=0)
