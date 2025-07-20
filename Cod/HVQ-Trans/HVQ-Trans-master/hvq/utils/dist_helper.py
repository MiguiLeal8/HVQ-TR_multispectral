import os
import subprocess

import torch
import torch.distributed as dist

def setup_distributed(backend="nccl", port=None):
    """Initialize distributed training environment.
    Supports both slurm and torch.distributed.launch
    see torch.distributed.init_process_group() for more details
    """
    # Check if GPUs are available
    num_gpus = torch.cuda.device_count()
    use_cuda = num_gpus > 0  # Determine if CUDA should be used based on GPU availability
    
    if not use_cuda:
        # Fallback to 'gloo' backend if no GPUs are present
        backend = "gloo"
    
    if "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        # Specify master port
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % (num_gpus if use_cuda else 1))
        os.environ["RANK"] = str(rank)
    else:
        # Define environment variables for torch.distributed.launch
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        os.environ["LOCAL_RANK"] = str(rank % (num_gpus if use_cuda else 1))
    
    # Set device only if GPUs are available
    if use_cuda:
        torch.cuda.set_device(rank % num_gpus)
    
    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )
    return rank, world_size