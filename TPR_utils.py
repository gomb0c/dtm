import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from node import Node
from vector_symbolic_utils import VectorSymbolicConverter, VectorSymbolicManipulator

class TPRConverter(VectorSymbolicConverter):
    def __init__(self, num_fillers, num_roles, d_filler=32, d_role=32) -> None:
        super().__init__()
        self.filler_emb = nn.Embedding(num_fillers, d_filler)
        self.role_emb = nn.Embedding(num_roles, d_role)

        self.filler_emb.requires_grad = False
        self.filler_emb.weight.requires_grad = False

        self.role_emb.requires_grad = False
        self.role_emb.weight.requires_grad = False

        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.orthogonal_(self.filler_emb.weight, gain=1)
        self.filler_emb.weight.data[0, :] = 0
        nn.init.orthogonal_(self.role_emb.weight, gain=1)

    def encode_tree_as_vector_symbolic(self, trees):
        '''
        Given a binary tree represented by a tensor, construct the TPR
        '''
        x = self.filler_emb(trees)
        return torch.einsum('brm,rn->bmn', x, self.role_emb.weight)
    
    def decode_vector_symbolic_to_tree(self, tpr_tensor, return_similarities=False):
        '''
        Given a TPR of dimension (B, D_{F}, D_{R}), unbind it into the underlying fillers
        Produces output of shape (B, N_{R], D_{F}}) if quantise_fillers is False
        Otherwise, produces output of shape (B, N_{R}, N_{F}) if quantise_fillers is True 
            (bins filler vectors into the N_{F} possible filler bins)
        '''
        unbinded = torch.einsum('bmn,rn->brm', tpr_tensor, self.role_emb.weight)
        if not return_similarities:
            return unbinded
        return torch.einsum('brm,fm->brf', unbinded, self.filler_emb.weight)

@torch.no_grad()
def build_E(role_emb):
    '''
    Build E matrices given the role embeddings (binary trees-only)
    '''
    d_role = role_emb.weight.size(1)
    E_l = role_emb.weight.new_zeros(d_role, d_role)
    E_r = role_emb.weight.new_zeros(d_role, d_role)
    def _add_to(mat, ind_from, ind_to):
        if ind_to >= role_emb.weight.size(0):
            return
        mat += torch.einsum('a,b->ab', role_emb.weight[ind_to], role_emb.weight[ind_from])
        _add_to(mat, ind_from*2+1, ind_to*2+1)
        _add_to(mat, ind_from*2+2, ind_to*2+2)
    _add_to(E_l, 0, 1)
    _add_to(E_r, 0, 2)
    E_l.requires_grad = False
    E_r.requires_grad = False
    return E_l, E_r

@torch.no_grad()
def build_D(role_emb):
    '''
    Build D matrices given the role embeddings (binary trees-only)
    '''
    d_role = role_emb.weight.size(1)
    D_l = role_emb.weight.new_zeros(d_role, d_role)
    D_r = role_emb.weight.new_zeros(d_role, d_role)
    def _add_to(mat, ind_from, ind_to):
        if ind_from >= role_emb.weight.size(0):
            return
        mat += torch.einsum('a,b->ab', role_emb.weight[ind_to], role_emb.weight[ind_from])
        _add_to(mat, ind_from*2+1, ind_to*2+1)
        _add_to(mat, ind_from*2+2, ind_to*2+2)
    _add_to(D_l, 1, 0)
    _add_to(D_r, 2, 0)
    D_l.requires_grad = False
    D_r.requires_grad = False
    return D_l, D_r

def DecodedTPR2Tree(decoded_tpr, eps=1e-2):
    contain_symbols = decoded_tpr.norm(p=2, dim=-1) > eps
    return torch.where(contain_symbols, decoded_tpr.argmax(dim=-1), 0)

# works for binary trees only
def Symbols2NodeTree(index_tree, i2v):
    def _traverse_and_detensorify(par, ind):
        if not index_tree[ind]:
            return par
        cur = Node(i2v[index_tree[ind]])
        if par:
            par.children.append(cur)
        if len(index_tree) > ind*2+1 and index_tree[ind*2+1]:
            # work on the left child
            _traverse_and_detensorify(cur, ind*2+1)
        if len(index_tree) > ind*2+2 and index_tree[ind*2+2]:
            # work on the right child
            _traverse_and_detensorify(cur, ind*2+2)
        return cur
    node_tree = _traverse_and_detensorify(None, 0)
    return node_tree

# example usage in main.py: BatchSymbols2NodeTree(fully_decoded, train_data.ind2vocab)
def BatchSymbols2NodeTree(decoded_tpr_batch, i2v):
    def s2nt(index_tree):
        return Symbols2NodeTree(index_tree, i2v)
    return list(map(s2nt, decoded_tpr_batch))

class TPRManipulator(VectorSymbolicManipulator): 
    def __init__(self, role_emb: torch.Tensor, num_ops: int=3, predefined_ops_random: bool=False) -> None: 
        if predefined_ops_random:
            d_role = role_emb.embedding_dim
            D_l = nn.Parameter(role_emb.weight.new_empty(d_role, d_role))
            D_r = nn.Parameter(role_emb.weight.new_empty(d_role, d_role))
            E_l = nn.Parameter(role_emb.weight.new_empty(d_role, d_role))
            E_r = nn.Parameter(role_emb.weight.new_empty(d_role, d_role))
            nn.init.kaiming_uniform_(D_l, a=math.sqrt(5))
            nn.init.kaiming_uniform_(D_r, a=math.sqrt(5))
            nn.init.kaiming_uniform_(E_l, a=math.sqrt(5))
            nn.init.kaiming_uniform_(E_r, a=math.sqrt(5))
        else:
            D_l, D_r = build_D(role_emb)
            E_l, E_r = build_E(role_emb)
        self.car_net = BBCarNet(D_l)
        self.cdr_net = BBCdrNet(D_r)
        self.cons_net = BBConsNet(E_l, E_r, role_emb.weight[0])
        self.num_ops = num_ops      
        
    def apply_car(self, tree_mem: torch.Tensor, weights: torch.Tensor) -> torch.Tensor: 
        return self.car_net(x=tree_mem, arg1_weight=weights)
    
    def apply_cdr(self, tree_mem: torch.Tensor, weights: torch.Tensor) -> torch.Tensor: 
        return self.cdr_net(x=tree_mem, arg1_weight=weights)
    
    def apply_cons(self, tree_mem: torch.Tensor, weights_l: torch.Tensor, 
                   weights_r: torch.Tensor, root_filler: torch.Tensor) -> torch.Tensor: 
        return self.cons_net(x=tree_mem, arg1_weight=weights_l, arg2_weight=weights_r, 
                             root_filler=root_filler)
      
class BBCarNet(nn.Module):
    def __init__(self, D_0) -> None:
        super().__init__()
        # hardcoded op
        self.car_weight = D_0

    def forward(self, x, arg1_weight):
        # batch, length, filler, role x batch, length
        arg1 = torch.einsum('blfr,bl->bfr', x, arg1_weight)
        # batch, filler, role_from x role_[t]o, role_from
        return torch.einsum('bfr,tr->bft', arg1, self.car_weight)

class BBCdrNet(nn.Module):
    def __init__(self, D_1) -> None:
        super().__init__()
        # hardcoded op
        self.cdr_weight = D_1

    def forward(self, x, arg1_weight):
        # batch, length, filler, role
        arg1 = torch.einsum('blfr,bl->bfr', x, arg1_weight)
        return F.linear(arg1, self.cdr_weight)

class BBConsNet(nn.Module):
    def __init__(self, E_0, E_1, root_role) -> None:
        super().__init__()
        # hardcoded op
        self.cons_l = E_0
        self.cons_r = E_1
        self.root_role = root_role

    def forward(self, x, arg1_weight, arg2_weight, root_filler):
        # batch, length, filler, role
        arg1 = torch.einsum('blfr,bl->bfr', x, arg1_weight)
        arg2 = torch.einsum('blfr,bl->bfr', x, arg2_weight)
        return F.linear(arg1, self.cons_l) + F.linear(arg2, self.cons_r) + torch.einsum('bf,r->bfr', root_filler, self.root_role)
