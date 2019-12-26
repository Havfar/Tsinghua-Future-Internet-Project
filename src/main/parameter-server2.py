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

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.distributed as dist
import torchvision.models as models
from torch.autograd import Variable


parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
parser.add_argument('--epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')


# def run(rank, world_size):
#     output = open("VGG19_PS_output.txt", "w")
#     args = parser.parse_args()

#     # model initiated with random weights
#     model = models.vgg19(pretrained=False)
    
#     # model_flat = flatten_all(model)
#     model_flat = flatten(model)
#     print("rank:", rank, "len model flat:", len(model_flat))
#     dist.broadcast(model_flat, 0)


#     # define loss function (criterion) and optimizer
#     criterion = nn.CrossEntropyLoss().cpu()

#     # NB! may want to remove
#     cudnn.benchmark = True

#     # Data loading code
#     train_transform = transforms.Compose(
#         [transforms.RandomCrop(32, padding=4),
#          transforms.RandomHorizontalFlip(),
#          transforms.ToTensor(),
#          transforms.Normalize((0.1307,), (0.3081,))])


#     # num workers kan forsøkes endres til = 0. Evt. les mer om dette.
#     # Drop last gjør at vi ikke får tull med at settet ikke kan deles på alle processene.
#     trainset = datasets.CIFAR10(root='./data', train=True, download=False, transform=train_transform)

#     # Tror denne fungerer slik at den passer på at hele datasettet blir fordelt på alle workers.
#     # I den forstand at hver worker velger tilfeldig data av datasettet ved hver epoch.
#     # Dette gjør at de ulike workerne får forskjellig data, men hele settet blir dekt ved hver epoch.
#     train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, num_replicas=world_size, rank=rank)

#     # Her bruker vi shuffle = false, som gjør at vi kan bruke en sampler, nemlig sampleren vi laget i linja over.
#     # Vet ikke om det er mer overhead å bruke sampler. Sampler velger visstnok et subset av training data for å trene på.

#     train_loader = torch.utils.data.DataLoader(trainset, batch_size=128 // world_size, pin_memory=True, drop_last=True ,shuffle=False,
#                                                num_workers=2, sampler=train_sampler)

#     # Her gjør vi noe transformering på verdiene. Ikke helt sikker på hvorfor vi normaliserer slik som vi gjør.
#     val_transform = transforms.Compose(
#         [transforms.ToTensor(),
#          transforms.Normalize((0.1307,), (0.3081,))])


#     valset = datasets.CIFAR10(root='./data', train=False, download=False, transform=val_transform)
#     val_loader = torch.utils.data.DataLoader(valset, batch_size=100, pin_memory=True, shuffle=False, num_workers=2)

#     time_cost = 0
#     for epoch in range(args.epochs):
#         print("Rank: ", rank, "starting epoch:", epoch)
#         dist.barrier()
#         t1 = time.time()
#         # for i, (input, target) in enumerate(val_loader):
#         for i in range(len(train_loader)):
#             model_flat = flatten(model)
#             # Todo endre rank
#             print("rank ", rank, "waiting for reduce of model", "i in len(trainloader):", i)
#             dist.reduce(model_flat, dst=rank, op=dist.ReduceOp.SUM)
#             model_flat.div_(2)

#             dist.broadcast(model_flat, src=rank)
#             unflatten(model, model_flat)

#         print("Rank:", rank, "calling dist.barrier()")
#         dist.barrier()
#         t2 = time.time()
#         time_cost += t2 - t1
#         model_flat.zero_()
#         loss = torch.FloatTensor([0])
#         dist.reduce(loss, dst=rank, op=dist.ReduceOp.SUM)
#         loss.div_(world_size)
#         dist.reduce(model_flat, dst=rank, op=dist.ReduceOp.SUM)
#         model_flat.div_(world_size)
#         #unflatten_all(model, model_flat)
#         unflatten(model, model_flat)

#         # evaluate on validation set
#         _, prec1 = validate(val_loader, model, criterion)
#         output.write('%d %3f %3f %3f\n' % (epoch, time_cost, loss.item(), prec1))
#         output.flush()

#     output.close()


def run(rank, world_size, pserver):
    output = open("VGG19_PS_output.txt", "w")
    args = parser.parse_args()
    current_lr = args.lr



    # model initiated with random weights
    model = models.vgg19(pretrained=False)
    
    model_flat = flatten(model)
    dist.broadcast(model_flat, src=pserver)

    print("rank:", rank, "len model flat:", len(model_flat))

    unflatten(model, model_flat)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cpu()

    # NB! may want to remove
    cudnn.benchmark = True

    # set optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=current_lr, momentum=0.9,
                                weight_decay=5e-4)  # weight_decay=0.0001)


    """

    # Data loading code from old run
    train_transform = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])


    trainset = datasets.CIFAR10(root='./data', train=True, download=False, transform=train_transform)

    size = world_size
    bsz = ceil(128 / float(size))
    partition_sizes = [1.0/ size for _ in range(size)]#[1.0 / size for _ in range(size)]

    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, num_replicas=world_size, rank=rank)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128 // world_size, pin_memory=True, drop_last=True ,shuffle=False,
                                               num_workers=2, sampler=train_sampler)


    """

    # if crash, can replace last line of transforms to
    #    transforms.Normalize((0.1307,), (0.3081,))])

    # Data loading code
    train_transform = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])


    # num workers kan forsøkes endres til = 0. Evt. les mer om dette.
    # Drop last gjør at vi ikke får tull med at settet ikke kan deles på alle processene.
    trainset = datasets.CIFAR10(root='./data', train=True, download=False, transform=train_transform)

    # Tror denne fungerer slik at den passer på at hele datasettet blir fordelt på alle workers.
    # I den forstand at hver worker velger tilfeldig data av datasettet ved hver epoch.
    # Dette gjør at de ulike workerne får forskjellig data, men hele settet blir dekt ved hver epoch.
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, num_replicas=world_size, rank=rank)

    # Her bruker vi shuffle = false, som gjør at vi kan bruke en sampler, nemlig sampleren vi laget i linja over.
    # Vet ikke om det er mer overhead å bruke sampler. Sampler velger visstnok et subset av training data for å trene på.

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128 // world_size, pin_memory=True, drop_last=True ,shuffle=False,
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
        # for i, (input, target) in enumerate(val_loader):

        train_sampler.set_epoch(epoch)


        loss = train(output, train_loader, model, criterion, optimizer, epoch, rank, world_size, pserver)


        print("Rank:", rank, "calling dist.barrier()")
        dist.barrier()
        t2 = time.time()
        time_cost += t2 - t1
        # model_flat.zero_()
        # loss = torch.FloatTensor([0])
        # dist.reduce(loss, dst=rank, op=dist.ReduceOp.SUM)
        # loss.div_(world_size)
        # dist.reduce(model_flat, dst=rank, op=dist.ReduceOp.SUM)
        # model_flat.div_(world_size)
        #unflatten_all(model, model_flat)
        # unflatten(model, model_flat)

        # evaluate on validation set

        if rank == pserver:
            output.write('Validate epoch', epoch)
            _, prec1 = validate(val_loader, model, criterion)
            output.write('%d %3f %3f %3f\n' % (epoch, time_cost, loss.item(), prec1))
            output.flush()

    output.close()

    ### run:

    for epoch in range(args.epochs):

        
        #model_flat = flatten_all(model)
        model_flat = flatten(model)

        #reduce loss and reduce the model

        # ingen grunn til å dele modellen her siden vi gjør det i hvert training steg
        #dist.reduce(torch.FloatTensor([loss]), pserver, op=dist.ReduceOp.SUM)
        #dist.broadcast(model_flat, pserver)

        # 
        
        #output.write('Epoch: %d  Time: %3f  Train_loss: %3f  Val_acc: %3f\n'%(epoch,time_cost,loss,prec1))


def train(output, train_loader, model, criterion, optimizer, epoch, rank, world_size, pserver):  # , model_l, model_r):
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()
    print("rank:", rank, "trainloader:", train_loader)

    # Run a batch
    for batch_idx, (input, target) in enumerate(train_loader):
        t1 = time.time()

        print("rank:", rank, "training batch:", batch_idx)
        input_var, target_var = Variable(input), Variable(target)

        # compute output
        model_output = model(input_var)
        loss = criterion(model_output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        # measure accuracy and record loss
        prec1 = accuracy(model_output.data, target, topk=(1,))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1[0], input.size(0))

        t2 = time.time()
        train_time = t2-t1

        # communicate
        model_flat = flatten(model)

        t3 = time.time()
        print("rank:", rank, "finished training in:", train_time, "starting reduce")
        dist.reduce(model_flat, dst=pserver, op=dist.ReduceOp.SUM)
        
        # sync all processes here after reducing -- so that we can divide in the coordinator and broadcast model
        dist.barrier()
        
        if rank == pserver:

            # average model
            model_flat.div_(world_size)
        
        dist.broadcast(model_flat, src=pserver)

        # sync all to make sure all have recieved updated averaged model
        dist.barrier()
        t4 = time.time()

        communication_time = t4-t3

        unflatten(model, model_flat)

        optimizer.step()

        output.write('%d %3f %3f %3f %3f %3f %3f %3f\n' % ("Rank: " + rank, "epoch: " + epoch, "batch", batch_idx, "train time cost: " + train_time, "communication time: " + communication_time, "loss: " + (loss.item()), "prec: "+ prec1))
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
