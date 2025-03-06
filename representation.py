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
    def decode_vector_symbolic_to_tree(self, repns: torch.Tensor, quantised_fillers: bool) -> torch.Tensor: 
        '''
        Converts the vector-symbolic representation of a tree to a tensor of fillers (if decode is False)
        of dimension (B, N_{R}, D_{F}), or fuzzy one-hot representation of the tree of dimension (B, N_{R}, N_{F})
        (if quantised_fillers is True)
        
        inputs:
            repns (torch.Tensor): corresponds to the TPR or VSA representation of shape
                (B, D_{F}, D_{R}) for TPRs, and (B, hypervec_dim) for VSAs
            quantised_fillers (bool): whether or not to quantise the $N_{R}$ fillers 
            into $N_{F}$ bins by taking the similarity between the filler values with the fillers in the vocabulary
        '''
        return
    
    