import os
import torch
import torch.distributed as dist
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F

from torch import optim

from torch.autograd import Variable


from math import ceil
from torchvision import transforms



from data_partition_helpers import DataPartitioner

DIST_BACKEND = "mpi"

def init_process(backend='mpi'):

    os.environ['MASTER_ADDR'] = 'nasp-cpu-01'
    os.environ['MASTER_PORT'] = '29500'

    # Initialize Process Group
    dist.init_process_group(backend=DIST_BACKEND)
        
    # get current process information
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    

    print("WORLD=", world_size)
    print("RANK=", rank)

def test_send_recv(rank):
    # dummy tensor
    d_tensor = torch.zeros(1)
    req1 = None
    req2 = None
    req3 = None
    test_tensor = torch.Tensor
    if rank == 0:
        req1 = dist.isend(tensor = d_tensor, dst = 1)
        req2 = dist.isend(tensor = d_tensor, dst = 2)
        print("Rank:", rank, "started sending data.")
        print("Rank:", rank, "data sent:", d_tensor)
        req1.wait()
        req2.wait()
    else:
        req3 = dist.irecv(tensor = d_tensor, src=0)
        print("Rank:", rank, "started receiving data.")
        req3.wait()
    print("Tensor:", d_tensor, "at rank:", rank)


def run(rank, size):
    """ Distributed Synchronous SGD Example """
    torch.manual_seed(1234)
    train_set, bsz = partition_dataset()
    model = models.vgg19()
    #model = model
#    model = model.cuda(rank)

    # Create optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    num_batches = ceil(len(train_set.dataset) / float(bsz))
    for epoch in range(1):
        epoch_loss = 0.0
        for data, target in train_set:
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            print("loss:", loss)
            print("loss.data.item():", loss.data.item())
            epoch_loss += loss.data.item()
            loss.backward()
            average_gradients(model)
            optimizer.step()
        print('Rank ',
              dist.get_rank(), ', epoch ', epoch, ': ',
              epoch_loss / num_batches)

def partition_dataset():
    print('==> Preparing data..')
    dataset = datasets.CIFAR10('./data', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))
                             ]))
    size = dist.get_world_size()
    bsz = ceil(128 / float(size))
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(partition,
                                         batch_size=bsz,
                                         shuffle=True)
    return train_set, bsz


def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data)
        param.grad.data /= size


def test_allreduce(rank):
    #group = dist.new_group([0, 1])
    tensor = torch.ones(1)
    dist.all_reduce(tensor = tensor)
    # print("Rank", rank, "has data", tensor[0])


# YOUR TRAINING CODE GOES HERE


if __name__ == "__main__":
    init_process()

    #Testing send/recv
    #test_send_recv(dist.get_rank())

    run(dist.get_rank(), dist.get_world_size())

    # calculate train_set and bsz
    # send to other nodes
    # works

    # works
    # test_allreduce(dist.get_rank())

    # receive train_set and bsz
    #else:


    # Create models
    # works
    # model = models.vgg19()
    # print("Rank", dist.get_rank(), "created model:", model)
    
    # define optimizer
    # works
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    # print("Rank", dist.get_rank(), "optimizer initiated:", optimizer)


    dist.barrier()
    dist.destroy_process_group()
