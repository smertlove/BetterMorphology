import os
import random
import numpy as np
import torch


# NOTE: Vibecoded with love and https://chat.deepseek.com/
def seed_everything(seed: int = 42):
    """
    Seed all random number generators for reproducibility.
    
    Args:
        seed: The seed value to use.
    """
    # Python's built-in random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # Python hash seed (affects set/dict iteration order in some cases)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    
    # PyTorch deterministic settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Enforce deterministic algorithms (PyTorch >= 1.8)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except AttributeError:
        pass
    # Required for some CUDA ops (e.g. atomicAdd) in older PyTorch
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    # Environment variables that affect some libraries
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"  # TensorFlow (if used alongside)


def worker_init_fn(worker_id: int):
    """
    Use with DataLoader(num_workers>0, worker_init_fn=worker_init_fn)
    to ensure each worker gets a different, reproducible seed.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
