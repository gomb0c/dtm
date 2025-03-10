import torch 

def cyclic_shift(x: torch.Tensor, shift: int) -> torch.Tensor: 
    return torch.roll(x, shifts=shift, dims=-1)

def bind_left(x: torch.Tensor) -> torch.Tensor: 
    return cyclic_shift(x, -1)

def bind_right(x: torch.Tensor) -> torch.Tensor: 
    return cyclic_shift(x, 1)

def unbind_left(x: torch.Tensor) -> torch.Tensor: 
    return bind_right(x) 

def unbind_right(x: torch.Tensor) -> torch.Tensor: 
    return bind_left(x)

def unbind_op(x: torch.Tensor, binding_op) -> torch.Tensor: 
    if binding_op is bind_left: 
        return unbind_left(x) 
    else: 
        if binding_op is not bind_right:
            raise NotImplementedError(f'Binding op is {binding_op}. Unbinding only implemented for LEFT/RIGHT')
        return unbind_right(x)
    
def bind_op(x: torch.Tensor, binding_op) -> torch.Tensor: 
    if binding_op is bind_left: 
        return bind_left(x)
    else: 
        if binding_op is not bind_right:
            raise NotImplementedError(f'Binding op is {binding_op}. Binding only implemented for LEFT/RIGHT')
        return bind_right(x)
        
def get_inv(binding_op) -> torch.Tensor: 
    if binding_op is bind_left: 
        return unbind_left
    else:
        if binding_op is not bind_right: 
            raise NotImplementedError(f'Binding op is {binding_op}. Inverse only implemented for LEFT/RIGHT')
        return unbind_right