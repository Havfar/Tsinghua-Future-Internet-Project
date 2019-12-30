# Cleaned up, and uses proper datapartitioner from pytorch documentation

import os
import shutil
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from data_partition_helpers import DataPartitioner
from math import ceil
from math import floor

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.distributed as dist
import torchvision.models as models
from torch.autograd import Variable


parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
parser.add_argument('--epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')



def run(rank, world_size, pserver):
    output = open("VGG19_PS_output_" + str(rank) + ".txt", "w")
    args = parser.parse_args()
    current_lr = args.lr



    # model initiated with random weights
    model = models.vgg19(pretrained=False)
    
    model_flat = flatten(model)
    dist.broadcast(model_flat, src=pserver)


    unflatten(model, model_flat)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cpu()

    # NB! may want to remove
    cudnn.benchmark = True

    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=current_lr, weight_decay=5e-4)
    # optimizer = torch.optim.SGD(model.parameters(), lr=current_lr, momentum=0.9,
    #                             weight_decay=5e-4)  # weight_decay=0.0001)


  
    # Data loading code
    train_transform = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])



    trainset = datasets.CIFAR10(root='./data', train=True, download=False, transform=train_transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, num_replicas=world_size, rank=rank)

  
    # batch_size_set = [0.15, 0.15, 0.15, 0.15, 0.10, 0.15, 0.15]
    # batch_size_set2 = [19, 19, 19, 19, 14, 19, 19]


    # Try with or without this
    """
    bsz = 128 / float(world_size)
    partition_sizes = [1.0 / world_size for _ in range(world_size)]
    partition = DataPartitioner(trainset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(partition,
                                         batch_size=bsz,
                                         shuffle=True)"""


    old_worldsize = 128 // world_size
    train_loader = torch.utils.data.DataLoader(trainset, batch_size= old_worldsize, pin_memory=True, drop_last=True ,shuffle=False,
                                               num_workers=2, sampler=train_sampler)

    # Her gjør vi noe transformering på verdiene. Ikke helt sikker på hvorfor vi normaliserer slik som vi gjør.
    val_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])


    valset = datasets.CIFAR10(root='./data', train=False, download=False, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=100, pin_memory=True, shuffle=False, num_workers=2)


    time_cost = 0
    for epoch in range(args.epochs):
        print("Rank: ", rank, "starting epoch:", epoch)
        dist.barrier()
        t1 = time.time()

        train_sampler.set_epoch(epoch)


        loss = train(output, train_loader, model, criterion, optimizer, epoch, rank, world_size, pserver)


        print("Rank:", rank, "calling dist.barrier()")
        dist.barrier()
        t2 = time.time()
        time_cost += t2 - t1

        # evaluate on validation set

        if rank == pserver:
            output.write('Validate epoch %s, \n' % (epoch))
            _, prec1 = validate(val_loader, model, criterion)
            output.write('epoch: %s, time: %s, loss: %s, accuracy: %s\n' % (str(epoch), str(time_cost), str(loss), str(prec1)))
            output.flush()

    output.close()


def train(output, train_loader, model, criterion, optimizer, epoch, rank, world_size, pserver):  # , model_l, model_r):
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()
    # print("rank:", rank, "trainloader:", train_loader)

    # Run a batch
    for batch_idx, (input, target) in enumerate(train_loader):
        
        optimizer.zero_grad()

        t1 = time.time()

        print("rank:", rank, "training batch:", batch_idx)
        input_var, target_var = Variable(input), Variable(target)

        # compute output
        model_output = model(input_var)
        loss = criterion(model_output, target_var)

        # compute gradient and do SGD step
        # optimizer.zero_grad() - moved to top of code
        loss.backward()

        # measure accuracy and record loss
        prec1 = accuracy(model_output.data, target, topk=(1,))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1[0], input.size(0))

        t2 = time.time()
        train_time = t2-t1

        # communicate
        model_flat = flatten(model)


        dist.barrier()
        t3 = time.time()
        print("rank:", rank, "finished training in:", train_time, "starting reduce")
        dist.all_reduce(model_flat, op=dist.ReduceOp.SUM)
        # sync all processes here after reducing -- so that we can divide in the coordinator and broadcast model
        dist.barrier()

        model_flat.div_(world_size)

        # dist.barrier()
        t4 = time.time()

        communication_time = t4-t3

        unflatten(model, model_flat)

        optimizer.step()

        output.write('Rank: %s, epoch: %s, batch_idx: %s, train-time: %s, com-time: %s, loss %s, prec: %s, \n' % ( str(rank), str(epoch), str(batch_idx), str(train_time), str(communication_time),  str(loss.item()), str(prec1[0].item())))
        output.flush()


    return losses.avg


def validate(val_loader, model, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (input, target) in enumerate(val_loader):
        input_var = Variable(input)
        target_var = Variable(target)

        # compute output
        with torch.no_grad():
            model_output = model(input_var)
            loss = criterion(model_output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(model_output.data, target, topk=(1,))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
    
    model.train()

    return losses.avg, top1.avg


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


def accuracy(model_output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = model_output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def flatten(model):
    vec = []
    for param in model.parameters():
        vec.append(param.data.view(-1))
    return torch.cat(vec)


def unflatten(model, vec):
    pointer = 0
    for param in model.parameters():
        num_param = torch.prod(torch.LongTensor(list(param.size())))
        param.data = vec[pointer:pointer + num_param].view(param.size())
        pointer += num_param


if __name__ == '__main__':
    dist.init_process_group('mpi')
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # 0 = nasp1 /i7
    # 1 = nasp2 /i3
    # 2 = nasp3 / i5
    pserver = 0
    run(rank, world_size, pserver)

    dist.destroy_process_group()
