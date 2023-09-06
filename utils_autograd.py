import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def set_grad_checkpoint(model):
    assert isinstance(model, nn.Module)

    def set_attr(module):
        module.grad_checkpointing = True
    
    model.apply(set_attr)


def auto_grad_checkpoint(module, *args, **kwargs):
    if getattr(module, 'grad_checkpointing', False):
        return checkpoint(module, *args, **kwargs)
    return module(*args, **kwargs)