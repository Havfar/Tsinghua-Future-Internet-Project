import torch
import torch.distributed as dist

DIST_BACKEND = "mpi"

# Initialize Process Group
dist.init_process_group(backend=DIST_BACKEND)
    
# get current process information
world_size = dist.get_world_size()
rank = dist.get_rank()

print("WORLD=", world_size)
print("RANK=", rank)

# YOUR TRAINING CODE GOES HERE

dist.barrier()
dist.destroy_process_group()
