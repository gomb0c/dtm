import torch 
import numpy as np
from absl.testing import absltest 

from VSA_utils import VSA
import constants.vsa_types as VSATypes

def get_vsa_instance(n_fillers: int, dim: int, vsa_type: VSATypes, bind_root: bool,
                 filler_weights: torch.Tensor = None, role_weights: torch.Tensor = None, 
                 strict_orth: bool=False) -> VSA:
    vsa = VSA(n_fillers, dim, vsa_type, bind_root, strict_orth)
    vsa._set_hypervecs(filler_weights, role_weights)
    return vsa

class VSAUtilsTestTreeHRR(absltest.TestCase): 
    def test_single_node_tree(self): 
        n_fillers = 2
        hypervec_dim = 7
        filler_weights = torch.Tensor([[1, 2, 4, 8, 16, 32, 64],
                                       [2, 4, 8, 16, 32, 64, 128]])
        role_weights = torch.randn_like(filler_weights)
        tree = torch.Tensor([2]) # note there is an offset here since a value of 0 indicates empty
        
        vsa = get_vsa_instance(n_fillers, hypervec_dim, VSATypes.HRR, 
                               bind_root=False, filler_weights=filler_weights, role_weights=role_weights)
        hrr_rep = vsa(tree.unsqueeze(0)) 
        np.testing.assert_allclose(hrr_rep.squeeze(0).numpy(), np.array([2, 4, 8, 16, 32, 64, 128]))
    
    def test_single_node_tree_root_role(self): 
        n_fillers = 2
        hypervec_dim = 7
        filler_weights = torch.Tensor([[1, 2, 4, 8, 16, 32, 64],
                                       [2, 4, 8, 16, 32, 64, 128]])
        role_weights = torch.randn(size=(2, hypervec_dim))
        role_weights = torch.concat([role_weights, 
                                     torch.Tensor([1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7]).unsqueeze(0)], dim=0)
        # root role has position 3 in the role_weights dict 
        # 
        tree = torch.Tensor([2])
        vsa = get_vsa_instance(n_fillers, hypervec_dim, VSATypes.HRR, 
                               bind_root=True, filler_weights=filler_weights, role_weights=role_weights)
        hrr_rep = vsa(tree.unsqueeze(0)) 
        # expected result is circ conv between filler_weights[1] and role_weights[positions.role_index]
        expected_result = (254/7)*torch.ones_like(filler_weights[0])
        np.testing.assert_allclose(hrr_rep.squeeze(0).numpy(), expected_result.numpy(), atol=1e-8, rtol=1e-6)
        
    def test_multinode_tree_balanced(self): 
        n_fillers = 3
        hypervec_dim = 4
        filler_weights = torch.Tensor([[1, 1, 1, 1], 
                                       [2, 3, 4, 5], 
                                       [1, 2, 3, 4]])
        role_weights = torch.Tensor([[1, 9, 8, 3],
                                     [2, 4, 8, 16]])
        tree = torch.Tensor([2, 3, 1])
        #       [2, 3, 4, 5]
        # [1, 2, 3, 4] [1, 1, 1, 1]
        vsa = get_vsa_instance(n_fillers, hypervec_dim, VSATypes.HRR, 
                               bind_root=False, filler_weights=filler_weights, role_weights=role_weights)
        hrr_rep = vsa(tree.unsqueeze(0))
        # expected result
        left_part = torch.Tensor([67, 52, 41, 50])
        right_part = torch.Tensor([30, 30, 30, 30])
        expected_result = left_part + right_part + filler_weights[1]
        np.testing.assert_allclose(hrr_rep.squeeze(0).numpy(), expected_result.numpy(), atol=1e-8, rtol=1e-6)
    
    def test_multinode_tree_balanced_root_role(self): 
        n_fillers = 3
        hypervec_dim = 4
        filler_weights = torch.Tensor([[1, 1, 1, 1], 
                                       [2, 3, 4, 5], 
                                       [1, 2, 3, 4]])
        role_weights = torch.Tensor([[1, 9, 8, 3],
                                     [2, 4, 8, 16],
                                     [1/2, 1/3, 1/4, 1/2]])
        tree = torch.Tensor([2, 3, 1])
        #       [2, 3, 4, 5]
        # [1, 2, 3, 4] [1, 1, 1, 1]
        vsa = get_vsa_instance(n_fillers, hypervec_dim, VSATypes.HRR, 
                               bind_root=True, filler_weights=filler_weights, role_weights=role_weights)
        hrr_rep = vsa(tree.unsqueeze(0))
        # expected result
        left_part = torch.Tensor([67, 52, 41, 50])
        right_part = torch.Tensor([30, 30, 30, 30])
        root_part = torch.Tensor([31/6, 65/12, 6, 67/12])
        expected_result = left_part + right_part + root_part
        np.testing.assert_allclose(hrr_rep.squeeze(0).numpy(), expected_result.numpy(), atol=1e-8, rtol=1e-6)
    
    def test_multinode_tree_unbalanced(self): 
        n_fillers = 5
        hypervec_dim = 3
        filler_weights = torch.Tensor([
            [1, 2, 3],
            [3, 4, 1],
            [2, 2, 1],
            [14, 21, 9],
            [9, 3, 2]
        ])
        role_weights = torch.Tensor([
            [1/2, 1/3, 1/4],
            [1/3, 1/3, -1/2]
        ])
        tree = torch.Tensor([1, 3, 5, 4, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0]) # 0 indicates empty
        #     1
        #   3   5
        #  4
        # 2
        vsa = get_vsa_instance(n_fillers, hypervec_dim, VSATypes.HRR, 
                               bind_root=False, filler_weights=filler_weights, role_weights=role_weights)
        # get hrr rep of tree rooted at 4
        ll_tree = torch.Tensor([14, 21, 9]) + torch.Tensor([17/6, 13/4, 31/12])
        
        # get hrr rep of tree rooted at 3
        l_tree = torch.Tensor([2, 2, 1]) + torch.Tensor([2641/144, 2971/144, 2604/144])
        
        # now get the hrr rep of the entire tree 
        left_part = torch.Tensor([38343/1728, 39514/1728, 38311/1728])
        right_part = torch.Tensor([13/6, 3, -17/6]) 
        expected_result = torch.Tensor([1, 2, 3]) + left_part + right_part
        
        hrr_rep = vsa(tree.unsqueeze(0))
        np.testing.assert_allclose(hrr_rep.squeeze(0).numpy(), expected_result.numpy(), atol=1e-8, rtol=1e-6)
        
    
    def test_multinode_tree_unbalanced_root_role(self): 
        pass 
    
class VSAUtilsTestBatchHRR(absltest.TestCase): 
    def test_single_elem_batch_no_empty_nodes(self): 
        pass 
    
    def test_single_elem_batch_empty_nodes(self): 
        pass
    
    def test_batched_no_empty_nodes(self): 
        pass 
    
    def test_batched_some_empty_nodes(self): 
        pass 
    
    
if __name__ == '__main__': 
    absltest.main()