'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import advertorch
import os
import argparse
from sklearn.metrics import classification_report
from torch.autograd import Variable
import utils
from datasets import cifar10

from wideresnet import WideResNet

import os.path as osp
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np


from attack import DeepFool

def get_embeds(model, loader):
    model = model.cuda().eval()
    full_embeds = []
    full_labels = []
    with torch.no_grad():
        for i, (feats, labels) in enumerate(loader):
            feats = feats[:100].cuda()
            full_labels.append(labels[:100].cpu().detach().numpy())
            embeds = model(feats)[2]
            full_embeds.append(F.normalize(embeds.detach().cpu()).numpy())
    # model = model.cpu()
    return np.concatenate(full_embeds), np.concatenate(full_labels)

def main():
    # Parse arguments.
    # args = parse_args()

    # Set device.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load dataset.
    train_loader, test_loader, class_names = cifar10.load_data('./')

    # Set a model.
    model = WideResNet(depth=28, num_classes=10, widen_factor=10)
    model = model.to(device)

    print('==> Resuming from checkpoint..')
    checkpoint = torch.load('checkpoint.pth')
    model.load_state_dict(checkpoint['net'])
    attack = DeepFool(model, 10)

    test(device, test_loader, model, attack)

def test(device, test_loader, model, attack):
    model.eval()

    output_list = []
    target_list = []
    running_loss = 0.0
    criterion = CWLoss(10)
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        # Forward processing.
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = attack(inputs, targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # Set data to calculate score.
        output_list += [int(o.argmax()) for o in outputs]
        target_list += [int(t) for t in targets]
        running_loss += loss.item()
    test_acc, test_loss = calc_score(output_list, target_list, running_loss, test_loader)

    print(test_acc)

    return test_acc, test_loss
def one_hot_tensor(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot

class CWLoss(nn.Module):
    def __init__(self, num_classes, margin=50, reduce=True):
        super(CWLoss, self).__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.reduce = reduce
        return


    def forward(self, logits, targets):
        """
        :param inputs: predictions
        :param targets: target labels
        :return: loss
        """

        onehot_targets = one_hot_tensor(targets, self.num_classes).cuda()

        self_loss = torch.sum(onehot_targets * logits.cuda(), dim=1)
        other_loss = torch.max(
            (1 - onehot_targets) * logits - onehot_targets * 1000, dim=1)[0]

        loss = -torch.sum(torch.clamp(self_loss - other_loss + self.margin, 0))

        if self.reduce:
            sample_num = onehot_targets.shape[0]
            loss = loss / sample_num

        return





def calc_score(output_list, target_list, running_loss, data_loader):
    # Calculate accuracy.
    result = classification_report(output_list, target_list, output_dict=True)
    acc = round(result['weighted avg']['f1-score'], 6)
    loss = round(running_loss / len(data_loader.dataset), 6)

    return acc, loss







if __name__ == "__main__":
    main()