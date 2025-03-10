from collections import deque
import torch 
import torch.nn as nn

import constants.vsa_types as VSATypes
import constants.positions as Positions
import ops.hrr_ops as hrr_ops
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
        elif vsa_type == VSATypes.HRR_NON_COMMUTATIVE: 
            self.bind_op = hrr_ops.non_commutative_binding 
            self.unbind_op = hrr_ops.non_commutative_unbinding
            self.generator = hrr_ops.generate_seed_vecs
        else: 
            raise NotImplementedError(f"{vsa_type} does not yet have implemented operations")

class VSAConverter(VectorSymbolicConverter): 
    def __init__(self, n_fillers: int, dim: int, 
                 vsa_operator: VSAOps, max_d: int=15,
                 bind_root: bool=False,
                 strict_orth: bool=False) -> None: 
        super().__init__() 
        self.strict_orth = strict_orth 
        self.filler_emb = nn.Embedding(num_embeddings=n_fillers,
                                        embedding_dim=dim)
        self.role_emb = nn.Embedding(num_embeddings=2 + int(bind_root), 
                                      embedding_dim=dim)
                
        self.hypervec_dim = dim
        self.n_hypervecs = 2 + int(bind_root) + n_fillers
        self.bind_root = bind_root
        self.vsa_operator = vsa_operator 
        self.n_roles = 2 + int(bind_root)
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
        self.root_role = nn.Parameter(self.role_emb.weight[Positions.ROOT_INDEX].unsqueeze(0), requires_grad=False) if self.bind_root else None
        
    def _set_hypervecs(self, filler_weights: torch.Tensor=None, role_weights: torch.Tensor=None) -> None:
        ''' For testing purposes '''
        if filler_weights is not None: 
            self.filler_emb.weight = nn.Parameter(filler_weights, requires_grad=False)
        if role_weights is not None: 
            self.role_emb.weight = nn.Parameter(role_weights, requires_grad=False)
            self.left_role = nn.Parameter(self.role_emb.weight[Positions.LEFT_INDEX].unsqueeze(0), requires_grad=False)
            self.right_role = nn.Parameter(self.role_emb.weight[Positions.RIGHT_INDEX].unsqueeze(0), requires_grad=False)
            self.root_role = nn.Parameter(self.role_emb.weight[Positions.ROOT_INDEX].unsqueeze(0), requires_grad=False) if self.bind_root else None

    
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
        return self.get_vsa_repn(trees)
    
    def get_vsa_repn(self, trees: torch.Tensor) -> torch.Tensor: 
        b_sz, max_nodes = trees.shape
        vsa_reps = torch.zeros(size=(b_sz, max_nodes, self.hypervec_dim), 
                               device=trees.device)
        
        for j in reversed(range(max_nodes)): 
            # get vocab indices for j-th node across match
            # shape (B, ) (0 => no node)
            vocab_idxs = trees[:, j]
            #print(f'Vocab idxs is {vocab_idxs}')
            
            # make mask of non-empty nodes
            valid_mask = (vocab_idxs > 0)
            
            if not torch.any(valid_mask): 
                continue 
            
            # filler vectors for valid batch entries at index j
            f = self.filler_emb.weight[vocab_idxs[valid_mask].long()] 
            #print(f'Shape of f is {f.shape}')
            
            vsa_reps_j = torch.zeros((b_sz, self.hypervec_dim), device=trees.device)
            vsa_reps_j[valid_mask] = f 
            print(f'At j, vocab_reps_j is {vsa_reps_j[valid_mask]}')
            #print(f'Filler is {f}\nhrr reps is {vsa_reps}\nvalid mask is {valid_mask}')
            
            # bind the root with a root role if applicable
            if j == 0 and self.bind_root: 
                vsa_reps_j[valid_mask] = self.vsa_operator.bind_op(self.root_role, f)
                print(f'J is {j}, result of binding {f} to root is {vsa_reps_j[valid_mask]}')
                print(f'After, vssa_reps_j is {vsa_reps_j[valid_mask]}')
            
            # left child 
            # ACCIDENTALLY BINDING J = 0 (ROOT TO LEFT ... ?)
            left_idx = 2*j + 1
            if left_idx < max_nodes: 
                left_child_mask = valid_mask & (trees[:, left_idx] > 0)
                if torch.any(left_child_mask): 
                    # convolve in batch -> (n_valid_children, hypervec_dim)
                    left_child_vecs = vsa_reps[left_child_mask, left_idx, :]
                    #print(f'LEFT CHILD INDEX {left_idx}')
                    conv_left = self.vsa_operator.bind_op(self.left_role, left_child_vecs)
                    #print(f'For j index {j}: Conv left between left_child_vecs at idx {left_idx} is {left_child_vecs} {conv_left}\n')
                    vsa_reps_j[left_child_mask] += conv_left 
                    #print(f'J is {j}, result of binding {left_child_vecs} to left is {vsa_reps_j[left_child_mask]}')
                    #print(f'After, vsa_reps_j is {vsa_reps_j[left_child_mask]}')
            # right child
            right_idx = 2*j + 2
            if right_idx < max_nodes: 
                right_child_mask = valid_mask & (trees[:, right_idx] > 0)
                if torch.any(right_child_mask): 
                    #print(f'RIGHT CHILD INDEX {right_idx}')
                    right_child_vecs = vsa_reps[right_child_mask, right_idx, :]
                    conv_right = self.vsa_operator.bind_op(self.right_role, right_child_vecs)
                    vsa_reps_j[right_child_mask] += conv_right
                    #print(f'For j index {j}: Conv right between right vecs at idx {right_idx} is {right_child_vecs} {conv_right}\n')
                    #print(f'After, vsa_reps_j is {vsa_reps_j[right_child_mask]}')
            vsa_reps[:, j, :] = vsa_reps_j
            #print(f'At iter {j}, we have {vsa_reps[:, j, :]}\n')
        
        #print(f'At the end, vsa reps is {vsa_reps}')
        return vsa_reps[:, 0, :] # corresponds to vsa rep of root
    
    
    def _decode_vsymbolic_level_order(self, vsa: torch.Tensor, eps: float=1e-10) -> torch.Tensor: 
        queue = deque()
        sims_list = []
        
        if self.bind_root: 
            vsa = self.vsa_operator.unbind_op(self.root_role, vsa)
        queue.append((vsa, 0))
        
        i = 0
        while queue: 
            current_vsa, d = queue.popleft()
            print(f'Current vsa is {current_vsa}')
            # process curr node 
            norm = torch.norm(current_vsa, keepdim=True)
            sim_mat = torch.cosine_similarity(
                current_vsa.unsqueeze(1), self.filler_emb.weight.unsqueeze(0), dim=-1
            ) # (B, 1, D), (1, N_{F}, D) -> (B, N_{F})

            sim_mat = torch.where(norm >= eps, sim_mat, torch.zeros_like(sim_mat)).unsqueeze(1) # (B, 1, N_{F})
            sims_list.append(sim_mat)
           
        
            if d < self.max_d - 1:
                # enqueue left and right 
                print(f'UNBINDING LEFT...')
                left_subtree = self.vsa_operator.unbind_op(self.left_role, current_vsa)
                print(f'FINISHED LEFT, which is {left_subtree}... UNBINDING RIGHT...')
                right_subtree = self.vsa_operator.unbind_op(self.right_role, current_vsa)
                
                queue.append((left_subtree, d+1))
                queue.append((right_subtree, d+1))
                print(f'Queue is {queue}')
        
        print(f'Sims list is {sims_list}')
        return torch.cat(sims_list, dim=1) # (B, 2**self.max_d-1, N_{F})
    
    
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
    