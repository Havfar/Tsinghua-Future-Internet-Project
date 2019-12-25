import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.models as models

from data_partition_helpers import DataPartitioner
from math import ceil
from torch import optim
from torch.autograd import Variable
from torchvision import transforms

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


# YOUR TRAINING CODE GOES HERE


def partition_dataset(includeTest):
    print('==> Preparing data..')
    # Where did we get our normalizing data from? They are quite different from main.py
    # Maybe we can improve our model by changing the transforms?
    dataset = datasets.CIFAR10('./data', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))]))
    test_set = None
    test_loader = None
    if includeTest:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)
        # test_set = datasets.CIFAR10('./data', train=False, download=True, transform=transforms.Compose(
        #     [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
    # changing size of dataset for testing on the servers before allocated time
    # as they run out of memory to hold all data during training
    # also speeds up testing sending data when training is faster
    # print("dataset:", dataset)
    # changing dataset.data to a slice of itself for testing on servers whilst cpu power is low (before our allocated time for testing)
    size = dist.get_world_size()
    bsz = ceil(128 / float(size))
    partition_sizes = [1.0 / size for _ in range(size)]#[1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())

    train_set = torch.utils.data.DataLoader(partition,
                                            batch_size=bsz,
                                            shuffle=True)
    return train_set, bsz, test_loader


def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data)
        param.grad.data /= size



def train(model, rank, optimizer, train_set, epoch, num_batches):
    dist.barrier()
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_set):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        # print("loss:", loss)
        # print("loss.data.item():", loss.data.item())
        epoch_loss += loss.data.item()
        loss.backward()
        average_gradients(model)
        optimizer.step()

        # Here begins the checking of accuracy and loss (new code)
        # inspired from main.py
        # Prints here are to figure out what the different stuff is
        # So that they can be troubleshot (correct english conjugation?)
        # print("output:", output)
        # print("output.max(1):", output.max(1))
        _, predicted = output.max(
            1)  # mulig bare .max? Vet ikke om output.max gir 2 return values for å fylle både _ og predicted?
        correct += predicted.eq(target).sum().item()
        # print("target:", target)
        # print("len(target)", len(target))
        total += len(target)
        # print("data:", data)
        print(batch_idx, "/", num_batches ," Rank:", rank, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
              % (epoch_loss / (batch_idx + 1), 100. * correct / total, correct, total))


    print('Rank ', rank, ', epoch ', epoch, ': ', epoch_loss / num_batches)

def test(model, test_set, rank, epoch, num_batches, output_file, training_time):
    # Thought: testing goes here, after all the training, since we don't call a train and a test function
    epoch_test_loss = 0
    test_correct = 0
    test_total = 0

    # Set model to evaluation mode [Introduced with accuracy, loss, testing]
    model.eval()
    with torch.no_grad():
        for test_batch_idx, (data, target) in enumerate(test_set):
            data, target = Variable(data), Variable(target)
            output = model(data)
            loss = F.nll_loss(output, target)

            epoch_test_loss += loss.data.item()
            _, predicted = output.max(1)
            test_total += len(target)
            test_correct += predicted.eq(target).sum().item()
            print("Rank:", rank, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                  % (epoch_test_loss / (test_batch_idx + 1), 100. * test_correct / test_total, test_correct,
                     test_total))
            """
            #Egen skrevet
            print("Rank:", dist.get_rank(), "loss:", epoch_test_loss/data+1, "Acc:", 100.*test_correct/test_total)
            """
        output_file.write('%d %3f %3f %3f \n' % (epoch, epoch_test_loss, test_correct / test_total, training_time))
        output_file.flush()
    print('Rank ',
          rank, ', epoch ', epoch, ': ',
          epoch_test_loss / num_batches)

def run(rank, validator):
    """ Distributed Synchronous SGD Example """
    torch.manual_seed(1234)
    output_file = open("VGG19_P2P_output.txt", "w")

    # Load train set and batch size - Extented in Accuracy, test, loss to include loading test_set
    inlcudeTestSet = (rank == validator)
    train_set, bsz, test_set = partition_dataset(inlcudeTestSet)
    model = models.vgg19()

    # Create optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    num_batches = ceil(len(train_set.dataset) / float(bsz))

    for epoch in range(60):
        # Set model to training mode [Introduced with accuracy, loss, testing]
        t1 = time.time()
        train(model, rank, optimizer, train_set, epoch, num_batches )
        t2 = time.time()
        print("Training time for epoch: ", epoch, " rank: ", rank, " time: ", t2-t1)
        if(rank == validator):
            test(model, test_set, rank, epoch, num_batches, output_file, t2-t1)

    output_file.close()

if __name__ == "__main__":
    init_process()

    # validator equal to nasp-cpu-01
    validator = 0
    run(dist.get_rank(), validator)
    dist.barrier()
    dist.destroy_process_group()

#############################################
#                                           #
# CODE USED FOR TESTING NO USE IN PROJECT   #
#                                           #
#############################################


"""
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
"""

"""
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
"""

"""
    def test_allreduce(rank):
    #group = dist.new_group([0, 1])
    tensor = torch.ones(1)
    dist.all_reduce(tensor = tensor)
    # print("Rank", rank, "has data", tensor[0])
    """