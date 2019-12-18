import os
import torch
import torch.distributed as dist

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
    d_tensor = torch.zeros(size = 1)
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


# YOUR TRAINING CODE GOES HERE


if __name__ == "__main__":
    init_process()
    test_send_recv(dist.get_rank())

    dist.barrier()
    dist.destroy_process_group()
