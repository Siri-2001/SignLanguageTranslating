import torch


def count_correct(y_h, y):
    y_h = torch.argmax(y_h.cpu(), dim=1)
    return sum(y_h == y)
