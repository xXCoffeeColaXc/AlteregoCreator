import torch
import torch.nn.functional as F


def adverarial_loss(output):
    #
    return -torch.mean(output)  # -log(D(x))


def classification_loss(logits, target):
    #return F.cross_entropy(classifier(real), labels)
    return F.binary_cross_entropy_with_logits(logits, target, size_average=False) / logits.size(0)


def reconstruction_loss(real, reconstructed):
    return torch.mean(torch.abs(real - reconstructed))  # L1 loss
