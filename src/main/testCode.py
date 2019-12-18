import os
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



def partition_dataset():
    print('==> Preparing data..')
    # Where did we get our normalizing data from? They are quite different from main.py
    # Maybe we can improve our model by changing the transforms?
    dataset = datasets.CIFAR10('./data', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))
                             ]))
    testset = datasets.CIFAR10('./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081))]))
    # changing size of dataset for testing on the servers before allocated time
    # as they run out of memory to hold all data during training
    # also speeds up testing sending data when training is faster
    print("dataset:", dataset)
    print("len(dataset):", len(dataset.data))
    # changing dataset.data to a slice of itself for testing on servers whilst cpu power is low (before our allocated time for testing)
    dataset.data = dataset.data[0:500]
    size = dist.get_world_size()
    bsz = ceil(128 / float(size))
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())

    train_set = torch.utils.data.DataLoader(partition,
                                         batch_size=bsz,
                                         shuffle=True)

    # Same procedure as above to generate test_set
    # Also sliced testset.data here to a smaller portion of itself
    # Only in order to speed up running the code on the servers during
    # Our initial testing of our code.
    testset.data = testset.data[0:500]
    test_size = dist.get_world_size()
    test_bsz = ceil(128 / float(test_size))
    partition_test_sizes = [1 / test_size for _ in range(test_size)]
    partition_test = DataPartitioner(testset, partition_test_sizes)
    partition_test = partition_test.use(dist.get_rank())
    test_set = torch.utils.data.DataLoader(partition_test, batch_size=test_bsz, shuffle=False)
    return train_set, bsz, test_set, test_bsz


def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data)
        param.grad.data /= size


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


def run(rank, size):
    """ Distributed Synchronous SGD Example """
    torch.manual_seed(1234)

    # Load train set and batch size - Extented in Accuracy, test, loss to include loading test_set
    train_set, bsz, test_set, test_bsz = partition_dataset()
    model = models.vgg19()

    # Create optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    num_batches = ceil(len(train_set.dataset) / float(bsz))
    for epoch in range(1):
        # Set model to training mode [Introduced with accuracy, loss, testing]
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
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

            # Here begins the checking of accuracy and loss (new code)
            # inspired from main.py
            # Prints here are to figure out what the different stuff is
            # So that they can be troubleshot (correct english conjugation?)
            print("output:", output)
            print("output.max(1):", output.max(1))
            _, predicted = output.max(1) # mulig bare .max? Vet ikke om output.max gir 2 return values for å fylle både _ og predicted?
            correct += predicted.eq(target).sum().item()
            print("target:", target)
            print("len(target)", len(target))
            total += len(target)
            print("Rank:", dist.get_rank, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (epoch_loss/(data+1), 100.*correct/total, correct, total))
            """
            # Egenskrevet
            print("Rank:", dist.get_rank(), "loss:", epoch_loss/data+1, "Acc:", 100.*correct/total)
            """
            
        print('Rank ',
              dist.get_rank(), ', epoch ', epoch, ': ',
              epoch_loss / num_batches)

        # Thought: testing goes here, after all the training, since we don't call a train and a test function
        epoch_test_loss = 0
        test_correct = 0
        test_total = 0

        # Set model to evaluation mode [Introduced with accuracy, loss, testing]
        model.eval()
        with torch.no_grad():
            for data, target in test_set:
                data, target = Variable(data), Variable(target)
                output = model(data)
                loss = F.nll_loss(output, target)

                epoch_test_loss += loss.data.item()
                _, predicted = output.max(1) # mulig bare .max? Vet ikke om output.max gir 2 return values for å fylle både _ og predicted?
                test_total += len(target) # egentlig targets.size i main.py, men det vil ikke fungere for oss tror jeg
                print(predicted.eq(target))
                test_correct += predicted.eq(target).sum().item()
                print("Rank:", dist.get_rank, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (epoch_test_loss/(data+1), 100.*test_correct/test_total, test_correct, test_total))
                """
                # Egen skrevet
                print("Rank:", dist.get_rank(), "loss:", epoch_test_loss/data+1, "Acc:", 100.*test_correct/test_total)"""



if __name__ == "__main__":
    init_process()
    run(dist.get_rank(), dist.get_world_size())
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