import abc
import torch
import torch.nn as nn

class VectorSymbolicConverter(nn.Module, metaclass=abc.ABCMeta):
    
    def forward(self, trees: torch.Tensor) -> torch.Tensor: 
        return self.encode_tree_as_vector_symbolic(trees)
    
    @abc.abstractmethod
    def encode_tree_as_vector_symbolic(self, trees: torch.Tensor) -> torch.Tensor: 
        '''
        Given a tensor representation of trees, converts it to the appropriate vector-symbolic
        representation
        
        inputs: 
            trees (torch.Tensor): corresponds to a tensor of dimension (B, 2**max_depth-1),
            Suppose val:= tree[i, j]. Then, the node at BFS position j is ind2vocab[val]
        ''' 
        return 
    
    @abc.abstractmethod 
    def decode_vector_symbolic_to_tree(self, repns: torch.Tensor, return_similarities: bool) -> torch.Tensor: 
        '''
        Converts the vector-symbolic representation of a tree to a tensor of fillers (if decode is False)
        of dimension (B, N_{R}, D_{F}), or matrix of distances of dimension (B, N_{R}, N_{F})
        (if return_similarities is True)
        
        inputs:
            repns (torch.Tensor): corresponds to the TPR or VSA representation of shape
                (B, D_{F}, D_{R}) for TPRs, and (B, hypervec_dim) for VSAs
            return_similarities (bool): whether or not to return the similarities between each of the $N_{R}$ 
            bound fillers with the N_{F} fillers in the dictionary
        '''
        return
    
    