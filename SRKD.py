'''
Implementation of "Knowledge Distillation via Softmax Regression Representation Learning"
ICLR 2021.
Unofficial Code
made by Hanbeen Lee
'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import model_dict
from dataset.cifar100 import *
from torchsummary import summary
from models.resnet import ResNet
import torch.optim as optim
import time

train_batch_size = 128
test_batch_size = 100
n_cls = 100
num_workers = 2
total_epoch = 240
learning_rate = 0.05
lr_decay_epoch = [150, 180, 210]
lr_decay_rate = 0.1
weight_decay = 5e-4
momentum = 0.9
dataset = 'cifar100'

model_t = model_dict['resnet32x4'](num_classes=100)
model_s = model_dict['resnet8x4'](num_classes=100)

path_t = './save/models/resnet32x4_cifar100_lr_0.05_decay_0.0005_trial_0/resnet32x4_best.pth'
trial = 0
r = 1
a = 1
b = 1
kd_T = 4


model_t.load_state_dict(torch.load(path_t)['model'])
model_t.eval()
teacher_classifier = model_t.fc
teacher_classifier.eval()

rand_value = torch.rand((2, 3, 32, 32))

feature, logit = model_s(rand_value, is_feat=True, preact=True)

train_loader, val_loader, n_data = get_cifar100_dataloaders(batch_size=train_batch_size,
                                                                        num_workers=num_workers,
                                                                        is_instance=True)
if torch.cuda.is_available():
    model_s.cuda()
    model_t.cuda()
    teacher_classifier.cuda()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(epoch, learning_rate, lr_decay_epochs,  optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(lr_decay_epochs))
    if steps > 0:
        new_lr = learning_rate * (lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


optimizer = optim.SGD(model_s.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
criterion_FM = nn.MSELoss()
criterion_SR = nn.KLDivLoss()
criterion_CE = nn.CrossEntropyLoss()

for epoch in range(1, total_epoch+1):
    adjust_learning_rate(epoch, learning_rate, lr_decay_epoch, optimizer)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()

    avg_loss_ce = 0
    avg_loss_fm = 0
    avg_loss_sr = 0
    count = 0

    for idx, data in enumerate(train_loader):
        input, target, index = data
        input = input.float()
        # target = target.float()
        optimizer.zero_grad()

        data_time.update(time.time() - end)
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        with torch.no_grad():
            feat_t, logit_t = model_t(input, is_feat=True, preact=True)
        feat_s, logit_s = model_s(input, is_feat=True, preact=True)

        loss_fm = criterion_FM(feat_s[-1], feat_t[-1])

        t_s_output = teacher_classifier(feat_s[-1])

        loss_sr = criterion_SR(F.log_softmax(t_s_output / kd_T, dim=1), F.softmax(logit_t / kd_T, dim=1)) * (kd_T ** 2)

        loss_ce = criterion_CE(F.softmax(logit_s), target)

        avg_loss_ce += loss_ce.item()
        avg_loss_fm += loss_fm.item()
        avg_loss_sr += loss_sr.item()
        count += 1
        loss = r * loss_ce + a * loss_fm + b * loss_sr
        loss.backward()
        optimizer.step()

    print(avg_loss_ce / count, avg_loss_fm / count, avg_loss_sr / count)













