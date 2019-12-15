"""run.py:"""
#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
from torch.distributions import transforms
from torch.multiprocessing import Process
import torchvision.datasets as datasets
import torchvision.models as models

#from src.main.data_partition_helpers import DataPartitioner
from data_partition_helpers import DataPartitioner

print("All imports completed")

def run(rank, size):

  #  dist.barrier() - Wait for porcesses
    print('==> Running ..')
    tensor = torch.zeros(1)
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        dist.send(tensor=tensor, dst=1)
    else:
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)
    print('Rank ', rank, ' has data ', tensor[0])
    # train_set, bsz = partition_dataset()
    # model = models.vgg19()
    # optimizer = optim.SGD(model.parameters(),
    #                       lr=0.01, momentum=0.5)
    #
    # num_batches = torch.ceil(len(train_set.dataset) / float(bsz))
    # for epoch in range(10):
    #     epoch_loss = 0.0
    #     for data, target in train_set:
    #         optimizer.zero_grad()
    #         output = model(data)
    #         loss = F.nll_loss(output, target)
    #         epoch_loss += loss.item()
    #         loss.backward()
    #         average_gradients(model)
    #         optimizer.step()
    #     print('Rank ', dist.get_rank(), ', epoch ',
    #           epoch, ': ', epoch_loss / num_batches)

""" Gradient averaging. """
def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size

def init_process(rank, size, fn, backend='mpi'):
    """ Initialize the distributed environment. """
    print('==> Initialize the distributed environment')
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

""" Partitioning CIFAR """
def partition_dataset():
    print('==> Preparing data..')
    dataset = datasets.CIFAR10('./data', train=True, download=True,
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

if __name__ == "__main__":
    size = 3
    processes = []
    for rank in range(size):
        print('==> Running rank: ' + rank +  " of " + size )
        p = Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
