import os
import torch
import torch.distributed as dist
import torchvision.datasets as datasets
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


# YOUR TRAINING CODE GOES HERE


if __name__ == "__main__":
    init_process()
    test_send_recv(dist.get_rank())


    # get train set and bsz
    if dist.get_rank() == 0:
        train_set, bsz = partition_dataset()
        print("train_set:", train_set, "bsz:", bsz)
        print("len(train_set):", len(train_set))

    dist.barrier()
    dist.destroy_process_group()
