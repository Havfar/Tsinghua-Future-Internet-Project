"""run.py:"""
#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
from torch import optim, nn
from torch.distributions import transforms
from torch.multiprocessing import Process
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F
from math import ceil
from torch.autograd import Variable

#from src.main.data_partition_helpers import DataPartitioner
from data_partition_helpers import DataPartitioner

from src.main.utils import progress_bar

print("All imports completed")

def run(rank, size):

  #  dist.barrier() - Wait for porcesses
    print('==> Running ..')
    # tensor = torch.zeros(1)
    # if rank == 0:
    #     tensor += 1
    #     # Send the tensor to process 1
    #     dist.send(tensor=tensor, dst=1)
    # else:
    #     # Receive tensor from process 0
    #     dist.recv(tensor=tensor, src=0)
    # print('Rank ', rank, ' has data ', tensor[0])


    """ Distributed Synchronous SGD Example """
    #torch.manual_seed(1234)
    train_set, bsz = partition_dataset()
    model = models.vgg19()
   # model = model
    #    model = model.cuda(rank)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    num_batches = ceil(len(train_set.dataset) / float(bsz))
    for epoch in range(1):
        epoch_loss = 0.0
        for data, target in train_set:
            data, target = Variable(data), Variable(target)
            #            data, target = Variable(data.cuda(rank)), Variable(target.cuda(rank))
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            epoch_loss += loss.data[0]
            loss.backward()
            average_gradients(model)
            optimizer.step()
        print('Rank ',
              dist.get_rank(), ', epoch ', epoch, ': ',
              epoch_loss / num_batches)

""" Gradient averaging. """
def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size

def init_process(rank, size, fn, backend='mpi'):
    """ Initialize the distributed environment. """
    print('==> Initialize the distributed environment')
    os.environ['MASTER_ADDR'] = 'nasp-cpu-01'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

""" Partitioning CIFAR """
def partition_dataset(train=True):
    print('==> Preparing data..')
    dataset = datasets.CIFAR10('./data', train=train, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))
                             ]))
    size = dist.get_world_size()
    bsz = 128 / float(size)
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(partition,
                                         batch_size=bsz,
                                         shuffle=True)
    return train_set, bsz

def train():
    for i in range(1):
        train_epoch(i)

# Training
def train_epoch(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    trainloader,bsz = partition_dataset(train)
    train_loss = 0
    correct = 0
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    testloader,bsz = partition_dataset(train=False)
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


if __name__ == "__main__":
    size = 3
    processes = []
    criterion = nn.CrossEntropyLoss()
    device = "cpu"
    model = models.vgg19()
    for rank in range(size):
        print('==> Running rank: ' + str(rank) +  " of " + str(size) )
        #p = Process(target=init_process, args=(rank, size, run))
        p = Process(target=train, args=(rank, model, device ))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Once training is complete, we can test the model
    for i in range(1):
        test(i)