from collections import deque
import torch 
import torch.nn as nn

import constants.vsa_types as VSATypes
import constants.positions as Positions
import ops.hrr_ops as hrr_ops
import ops.perm_ops as perm_ops
from vector_symbolic_utils import VectorSymbolicConverter, VectorSymbolicManipulator

'''
To think about: maybe orthogonalise the hypervec space based on the number of actual fillers (not including empty filler?)
'''
class VSAOps(): 
    def __init__(self, vsa_type: VSATypes=VSATypes.HRR_STANDARD) -> None: 
        if vsa_type == VSATypes.HRR_STANDARD: 
            self.bind_op = hrr_ops.standard_binding
            self.unbind_op = hrr_ops.standard_unbinding 
            self.generator = hrr_ops.generate_seed_vecs
        else: 
            raise NotImplementedError(f"{vsa_type} does not yet have implemented operations")

class VSAConverter(VectorSymbolicConverter): 
    def __init__(self, n_fillers: int, dim: int, 
                 vsa_operator: VSAOps, max_d: int=15,
                 strict_orth: bool=False) -> None: 
        super().__init__() 
        self.strict_orth = strict_orth 
        self.filler_emb = nn.Embedding(num_embeddings=n_fillers,
                                        embedding_dim=dim)
        self.role_emb = nn.Embedding(num_embeddings=2, 
                                      embedding_dim=dim)
                
        self.hypervec_dim = dim
        self.n_hypervecs = 2 + n_fillers
        self.vsa_operator = vsa_operator 
        self.n_roles = 2
        self.n_fillers = n_fillers
        self.max_d = max_d
        
        # initialise all seed hypervectors according to https://arxiv.org/pdf/2109.02157
        # use unitary seed hypervectors 
        
        self.filler_emb.requires_grad = False 
        self.filler_emb.weight.requires_grad = False 
        self.role_emb.requires_grad = False
        self.role_emb.weight.requires_grad = False 
        
        print(f'Bind op is {self.vsa_operator.bind_op}')
        
        self.init_hypervecs()
        self.left_role = nn.Parameter(self.role_emb.weight[Positions.LEFT_INDEX].unsqueeze(0), requires_grad=False)
        self.right_role = nn.Parameter(self.role_emb.weight[Positions.RIGHT_INDEX].unsqueeze(0), requires_grad=False)

    def _set_hypervecs(self, filler_weights: torch.Tensor=None, role_weights: torch.Tensor=None) -> None:
        ''' For testing purposes '''
        if filler_weights is not None: 
            self.filler_emb.weight = nn.Parameter(filler_weights, requires_grad=False)
        if role_weights is not None: 
            self.role_emb.weight = nn.Parameter(role_weights, requires_grad=False)
            self.left_role = nn.Parameter(self.role_emb.weight[Positions.LEFT_INDEX].unsqueeze(0), requires_grad=False)
            self.right_role = nn.Parameter(self.role_emb.weight[Positions.RIGHT_INDEX].unsqueeze(0), requires_grad=False)

    def init_hypervecs(self) -> None:
        print(f'INITIALISING HYPERVECS')
        hypervecs = self.vsa_operator.generator(n_vecs=self.n_hypervecs,
                                                   dims=self.hypervec_dim,
                                                   strict_orth=self.strict_orth)
        self.role_emb.weight = nn.Parameter(hypervecs[:self.n_roles], requires_grad=False)
        self.filler_emb.weight = nn.Parameter(hypervecs[self.n_roles:], requires_grad=False) 
        self.filler_emb.weight[0] = 0 
        
    def encode_stree(self, trees: torch.Tensor) -> torch.Tensor: 
        '''
        inputs: 
            trees (torch.Tensor) corresponds to a tensor of dimension (B, 2**max_depth-1),
                Suppose val:= tree[i, j]. Then, the node at BFS position j is ind2vocab[val]
        '''
        b_sz = trees.shape[0]
        vsa_reps = torch.zeros(size=(b_sz, self.hypervec_dim), device=trees.device)
        print(f'VSA reps {vsa_reps}')
        
        def recursive_encode(node_idx: int, role: torch.Tensor, vsa_reps: torch.Tensor) -> None:     
            if node_idx < 2**self.max_d - 1: 
                filler_idxs = trees[:, node_idx].long()
                print(f'Filler idxs is {filler_idxs}')
                fillers = self.filler_emb.weight[filler_idxs]
                print(f'Node idx is {node_idx}')
                if node_idx == 0:
                    vsa_reps += fillers 
                    print(f'VSA reps is {vsa_reps}')
                    new_role = role
                else: 
                    if node_idx % 2 == 0: # right node
                        print(f'IN RIGHT NODE WITH NODE IDX {node_idx}\n')
                        if node_idx == 2:
                            new_role = self.right_role 
                        else: 
                            new_role = self.vsa_operator.bind_op(perm_ops.cyclic_shift(self.right_role, 1), role)
                        print(f'')
                        vsa_reps += self.vsa_operator.bind_op(fillers, new_role)
                        print(f'Conv between new role {new_role} and fillers {fillers} is {self.vsa_operator.bind_op(fillers, new_role)}\n')
                    else: 
                        print(f'IN LEFT NODE WITH NODE IDX {node_idx}\n')
                        if node_idx == 1:
                            new_role = self.left_role 
                        else:
                            new_role = self.vsa_operator.bind_op(perm_ops.cyclic_shift(self.left_role, 1), role)
                        vsa_reps += self.vsa_operator.bind_op(fillers, new_role)
                        print(f'Conv between new role {new_role} and fillers {fillers} is {self.vsa_operator.bind_op(fillers, new_role)}\n')
                recursive_encode(node_idx+1, new_role, vsa_reps)
            else: 
                return 
            
        recursive_encode(0, hrr_ops.get_conv_identity(d=self.hypervec_dim).unsqueeze(0), vsa_reps)
        return vsa_reps
        
        
    
    def decode_vsymbolic(self, vsa: torch.Tensor, return_distances: bool=True,
                                       eps: float=1e-10) -> torch.Tensor: 
        ''''
        Given a vsa, decodes the vsa into either 1) a set of N_{R} D-dimensional filler embeddings (if quantise_fillers is False),
        where result[i] is a D-dimensional filler embedding for the i-th position in the tree (where the tree is traversed in a BFS)
        or 2) a tensor of dimension (n_nodes, N_{F}), where result[i][j] desnotes the cosine similarities between the filler bound 
        to position i and the j-th filler
        
        Inputs: 
            vsa (torch.Tensor): a tensor of dimension (B, D) representing a VSA representation of a recursively defined binary tree
            return_distances (bool): if true, return a tensor of dimension (n_nodes, N_{F}) representing the distances between the 
            filler bound to role i, and each of the N_{F} fillers
        Returns
            final (torch.Tensor) tensor of dimension (B, 2**max_depth-1, N_{F})
        '''
        
        print(f'Input vsa is {vsa} with shape {vsa.shape}')
        
        if not return_distances:
            raise ValueError(f'For VSAs, we cannot directly extract out the filler embedding!')
        return self._decode_vsymbolic_level_order(vsa, eps=eps)
        
          
        
class VSAManipulator(VectorSymbolicManipulator): 
    def __init__(self, vsa_ops: VSAOps, left_role: torch.Tensor, right_role: torch.Tensor, root_role: torch.Tensor) -> None: 
        super().__init__()
        self.cons_net = VSAConsNet(vsa_ops, left_role, right_role, root_role)
        self.car_net = VSACarNet(vsa_ops, left_role)
        self.cdr_net = VSACdrNet(vsa_ops, right_role)
    
    def apply_car(self, tree_mem, weights):
        return self.car_net(tree_mem, weights)
    
    def apply_cdr(self, tree_mem, weights):
        return self.car_net(tree_mem, weights)
    
    def apply_cons(self, tree_mem, weights_l, weights_r, root_filler):
        return self.car_net(tree_mem, weights_l, weights_r, root_filler)
                
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
    