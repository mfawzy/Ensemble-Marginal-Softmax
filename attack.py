import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
import copy
import numpy as np

import torch
# from attacks import Attack


import os
import torch

class Attack(object):
    """Base class for attacks
    Arguments:
        object {[type]} -- [description]
    """
    def __init__(self, attack_type, target_cls, img_type='float'):
        self.attack_name = attack_type
        self.target_cls = target_cls

        self.training = target_cls.training
        self.device = next(target_cls.parameters()).device

        self.mode = img_type

    def forward(self, *args):
        """Call adversarial examples
        Should be overridden by all attakc classes
        """
        raise NotImplementedError

    def inference(self, save_path, file_name, data_loader):
        """[summary]
        Arguments:
            save_path {[type]} -- [description]
            data_loader {[type]} -- [description]
        """
        self.target_cls.eval()

        adv_list = []
        label_list = []

        correct = 0
        accumulated_num = 0.
        total_num = len(data_loader)

        for step, (imgs, labels) in enumerate(data_loader):
            adv_imgs, labels = self.__call__(imgs, labels)

            adv_list.append(adv_imgs.cpu())
            label_list.append(labels.cpu())

            accumulated_num += labels.size(0)

            if self.mode.lower() == 'int':
                adv_imgs = adv_imgs.float()/255.

            outputs = self.target_cls(adv_imgs)
            _, predicted = torch.max(outputs, 1)
            correct += predicted.eq(labels).sum().item()

            acc = 100 * correct / accumulated_num

            print('Progress : {:.2f}% / Accuracy : {:.2f}%'.format(
                (step+1)/total_num*100, acc), end='\r')

        adversarials = torch.cat(adv_list, 0)
        y = torch.cat(label_list, 0)

        os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(save_path, file_name)
        torch.save((adversarials, y), save_path)
        print("\n Save Images & Labels")

    def __call__(self, *args, **kwargs):
        self.target_cls.eval()
        adv_examples = self.forward(*args, **kwargs)

        if self.mode.lower() == 'int':
            adv_examples = (adv_examples*255).type(torch.uint8)

        return adv_examples

class DeepFool(Attack):
    """Reproduce DeepFool
    in the paper 'DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks'
    Arguments:
        target_cls {nn.Module} -- Target classifier to fool
        n_iters {n_iters} -- Step size
    """
    def __init__(self, target_cls, iters):
        super(DeepFool, self).__init__("DeepFool", target_cls)
        self.n_iters = iters

    def forward(self, imgs, labels):
        imgs = imgs.to(self.device)
        labels = labels.to(self.device)

        for idx, img in enumerate(imgs):
            img = img.unsqueeze(0)
            img.requires_grad = True

            output = self.target_cls(img)[0]

            _, first_predict = torch.max(output, 0)
            first_max = output[first_predict]
            grad_first = torch.autograd.grad(first_max, img)[0]

            num_classes = len(output)

            for _ in range(self.n_iters):
                img.requires_grad = True
                output = self.target_cls(img)[0]
                _, predict = torch.max(output, 0)

                if predict != first_predict:
                    img = torch.clamp(img, min=0, max=1).detach()
                    break

                r = None
                min_value = None

                for k in range(num_classes):
                    if k == first_predict:
                        continue

                    k_max = output[k]
                    grad_k = torch.autograd.grad(
                        k_max, img, retain_graph=True, create_graph=True)[0]

                    prime_max = k_max - first_max
                    grad_prime = grad_k - grad_first
                    value = torch.abs(prime_max)/torch.norm(grad_prime)

                    if r is None:
                        r = (torch.abs(prime_max)/(torch.norm(grad_prime)**2))*grad_prime
                        min_value = value
                    else:
                        if min_value > value:
                            r = (torch.abs(prime_max)/(torch.norm(grad_prime)**2))*grad_prime
                            min_value = value

                img = torch.clamp(img+r, min=0, max=1).detach()

            imgs[idx:idx+1, :, :, :] = img

        return imgs


def local_attack(model,  img, label, eps, attack_type, iters, criterion, random_restart=True):

    if random_restart:
        adv = img.detach() + torch.zeros_like(img).uniform_(-eps, eps)
    adv.requires_grad = True

    if attack_type == 'fgsm':
        iterations = 1
    else:
        iterations = iters

    if attack_type == 'pgd':
        step = 2 / 255
    else:
        step = eps / iterations
    adv_noise = 0
    for j in range(iterations):

        # adv_out = model(adv)
        f, _, adv_out = model(adv)
        loss = criterion(adv_out, label)
        loss.backward()

        if attack_type == 'mifgsm':
            adv.grad = adv.grad / torch.mean(torch.abs(adv.grad), dim=(1, 2, 3), keepdim=True)
            adv_noise = adv_noise + adv.grad
        else:
            adv_noise = adv.grad

        adv.data = adv.data + step * adv_noise.sign()


        # Projection
        if attack_type == 'pgd':
            adv.data = torch.min(torch.max(adv.data, img - eps), img + eps)
        adv.data.clamp_(0.0, 1.0)

        adv.grad.data.zero_()

    return adv.detach()
