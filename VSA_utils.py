import torch 
import torch.nn as nn

import constants.vsa_types as VSATypes
import constants.positions as Positions
import hrr_ops as hrr_ops

class VSAOps(): 
    def __init__(self, vsa_type: VSATypes=VSATypes.HRR) -> None: 
        if vsa_type == VSATypes.HRR: 
            self.bind_op = hrr_ops.circular_conv
            self.unbind_op = hrr_ops.circular_conv 
            self.inverse_op = hrr_ops.get_inv
            self.generator = hrr_ops.generate_seed_vecs
        else: 
            raise NotImplementedError(f"{vsa_type} does not yet have implemented operations")

    
class VSA(nn.Module): 
    def __init__(self, n_fillers: int, dim: int, 
                 vsa_operator: VSAOps,
                 bind_root: bool=False,
                 strict_orth: bool=False) -> None: 
        super().__init__() 
        self.strict_orth = strict_orth 
        self.filler_dict = nn.Embedding(num_embeddings=n_fillers,
                                        embedding_dim=dim)
        self.role_dict = nn.Embedding(num_embeddings=2 + int(bind_root), 
                                      embedding_dim=dim)
                
        self.hypervec_dim = dim
        self.n_hypervecs = 2 + int(bind_root) + n_fillers
        self.bind_root = bind_root
        self.vsa_operator = vsa_operator 
        self.n_roles = 2 + int(bind_root)
        
        # initialise all seed hypervectors according to https://arxiv.org/pdf/2109.02157
        # use unitary seed hypervectors 
        
        self.filler_dict.requires_grad = False 
        self.filler_dict.weight.requires_grad = False 
        self.role_dict.requires_grad = False
        self.role_dict.weight.requires_grad = False 
        
        self.init_hypervecs()
        self.left_role = nn.Parameter(self.role_dict.weight[Positions.LEFT_INDEX], requires_grad=False)
        self.right_role = nn.Parameter(self.role_dict.weight[Positions.RIGHT_INDEX], requires_grad=False)
        self.root_role = nn.Parameter(self.role_dict.weight[Positions.ROOT_INDEX], requires_grad=False) if self.bind_root else None
    
    def _set_hypervecs(self, filler_weights: torch.Tensor=None, role_weights: torch.Tensor=None) -> None:
        ''' For testing purposes '''
        if filler_weights is not None: 
            self.filler_dict.weight = nn.Parameter(filler_weights, requires_grad=False)
        if role_weights is not None: 
            self.role_dict.weight = nn.Parameter(role_weights, requires_grad=False)
            self.left_role = nn.Parameter(self.role_dict.weight[Positions.LEFT_INDEX].unsqueeze(0), requires_grad=False)
            self.right_role = nn.Parameter(self.role_dict.weight[Positions.RIGHT_INDEX].unsqueeze(0), requires_grad=False)
            self.root_role = nn.Parameter(self.role_dict.weight[Positions.ROOT_INDEX].unsqueeze(0), requires_grad=False) if self.bind_root else None
    
    def init_hypervecs(self) -> None:
        hypervecs = self.vsa_operator.generator(n_vecs=self.n_hypervecs,
                                                   dims=self.hypervec_dim,
                                                   strict_orth=self.strict_orth)
        self.role_dict.weight = nn.Parameter(hypervecs[:self.n_roles], requires_grad=False)
        self.filler_dict.weight = nn.Parameter(hypervecs[self.n_roles, :], requires_grad=False)
    
    def forward(self, trees: torch.Tensor) -> torch.Tensor: 
        '''
        inputs: 
            trees (torch.Tensor) corresponds to a tensor of dimension (B, 2**max_depth-1),
                Suppose val:= tree[i, j]. Then, the node at BFS position j is ind2vocab[val]
        '''
        return self.get_vsa_repn(trees)
    
    def get_vsa_repn(self, trees: torch.Tensor) -> torch.Tensor: 
        b_sz, max_nodes = trees.shape
        vsa_reps = torch.zeros(size=(b_sz, max_nodes, self.hypervec_dim), 
                               device=trees.device)
        
        for j in reversed(range(max_nodes)): 
            # get vocab indices for j-th node across match
            # shape (B, ) (0 => no node)
            vocab_idxs = trees[:, j]
            
            # make mask of non-empty nodes
            valid_mask = (vocab_idxs > 0)
            
            if not torch.any(valid_mask): 
                continue 
            
            # filler vectors for valid batch entries at index j
            f = self.filler_dict.weight[vocab_idxs[valid_mask].long() - 1] 
            # note that we offset by 1 due to the fact that vocab_idx of 0 is used 
            # by the authors of this repo to indicate that a position is empty
            #print(f'Shape of f is {f.shape}')
            
            vsa_reps_j = torch.zeros((b_sz, self.hypervec_dim), device=trees.device)
            vsa_reps_j[valid_mask] = f 
            #print(f'Filler is {f}\nhrr reps is {vsa_reps}\nvalid mask is {valid_mask}')
            
            # bind the root with a root role if applicable
            if j == 0 and self.bind_root: 
                vsa_reps_j[valid_mask] = self.vsa_operator.bind_op(self.root_role, f)
                #print(f'J == 0 and circular conv is {vsa_reps_j[valid_mask]}\nRoot role is {self.root_role}, f is {f}')
            
            # left child 
            left_idx = 2*j + 1
            if left_idx < max_nodes: 
                left_child_mask = valid_mask & (trees[:, left_idx] > 0)
                if torch.any(left_child_mask): 
                    # convolve in batch -> (n_valid_children, hypervec_dim)
                    left_child_vecs = vsa_reps[left_child_mask, left_idx, :]
                    conv_left = self.vsa_operator.bind_op(self.left_role, left_child_vecs)
                    #print(f'Hrr reps before is {vsa_reps_j[left_child_mask]}')
                    vsa_reps_j[left_child_mask] += conv_left 
                    #print(f'Left child is {left_child_vecs}\nconv result is {conv_left}\nleft role is {self.left_role}')
                    #print(f'Hrr reps after is {vsa_reps_j[left_child_mask]}\n')
            # right child
            right_idx = 2*j + 2
            if right_idx < max_nodes: 
                right_child_mask = valid_mask & (trees[:, right_idx] > 0)
                if torch.any(right_child_mask): 
                    right_child_vecs = vsa_reps[right_child_mask, right_idx, :]
                    conv_right = self.vsa_operator.bind_op(self.right_role, right_child_vecs)
                    vsa_reps_j[right_child_mask] += conv_right
                    #print(f'Right child is {right_child_vecs}, j is {vsa_reps_j[right_child_mask]}, right role is {self.right_role}')
            
            vsa_reps[:, j, :] = vsa_reps_j
        
        return vsa_reps[:, 0, :] # corresponds to vsa rep of root
                
class VSAConsNet(nn.Module): 
    def __init__(self, vsa_ops: VSAOps, left_role: torch.Tensor, right_role: torch.Tensor, 
                 root_role: torch.Tensor) -> None: 
        self.vsa_ops = vsa_ops
        self.left_role = left_role 
        self.right_role = right_role 
        self.root_role = root_role 
    
    def forward(self, tree_mem: torch.Tensor, weights_left: torch.Tensor, 
                weights_right: torch.Tensor, root_filler) -> torch.Tensor: 
        '''
        inputs:  
            tree_mem (torch.Tensor) of dimension (B, T, D), where T denotes 
                number of steps/memory cells, and D denotes hypervec dim
            weights_left (torch.Tensor) of dimension (B, T)
            weights_right (torch.Tensor) of dimension (B, T)
            root_filler (torch.Tensor) of dimension (B, D)
        '''
        left_st = torch.einsum('btd,bt->bd', tree_mem, weights_left)
        right_st = torch.einsum('btd,bt->bd', tree_mem, weights_right)
        return (self.vsa_ops.bind(left_st, self.left_role) + 
                self.vsa_ops.bind(right_st, self.right_role) + self.vsa_ops.bind(root_filler, self.root_role))
    
    
class VSACarNet(nn.Module): # left 
    def __init__(self, vsa_ops: VSAOps, left_role: torch.Tensor) -> None: 
        self.vsa_ops = vsa_ops 
        self.left_role = left_role 
    
    def forward(self, tree_mem: torch.Tensor, weights) -> torch.Tensor: 
        weighted_tree = torch.einsum('btd,bt->bd', tree_mem, weights)
        return self.vsa_ops.bind(weighted_tree, self.vsa_ops.inverse_op(self.left_role))
    
class VSACdrNet(nn.Module): # right
    def __init__(self, vsa_ops: VSAOps, right_role: torch.Tensor) -> None: 
        self.vsa_ops = vsa_ops 
        self.right_role = right_role 
    
    def forward(self, tree_mem: torch.Tensor, weights) -> torch.Tensor: 
        weighted_tree = torch.einsum('btd,bt->bd', tree_mem, weights)
        return self.vsa_ops.bind(weighted_tree, self.vsa_ops.inverse_op(self.right_role)) 
    