import math
import torch 
import numpy as np
import sys 
from absl.testing import absltest 

from VSA_utils import VSAConverter, VSAOps
import constants.vsa_types as VSATypes

torch.manual_seed(1234)
np.random.seed(1234)

def get_vsa_instance(n_fillers: int, dim: int, vsa_type: VSATypes, bind_root: bool,
                     filler_weights: torch.Tensor = None, role_weights: torch.Tensor = None, 
                     max_d: int=3, strict_orth: bool=False) -> VSAConverter:
    vsa_converter = VSAConverter(n_fillers, dim, VSAOps(vsa_type), max_d, bind_root, strict_orth)
    if filler_weights is not None and role_weights is not None: 
        print(f'Setting filler weights and role weights to {filler_weights}, {role_weights}')
        vsa_converter._set_hypervecs(filler_weights, role_weights)
    return vsa_converter


class VSAConverterInitialise(absltest.TestCase):
    def test_init_seed_vectors(self): 
        vsa_instance = get_vsa_instance(n_fillers=1000, dim=1024, 
                                        vsa_type=VSATypes.HRR_STANDARD, bind_root=False, 
                                        filler_weights=None, role_weights=None, 
                                        max_d=3, strict_orth=False)
        filler_embs = vsa_instance.filler_emb.weight 
        role_embs = vsa_instance.role_emb.weight
        
        # check that the complex magnitude is equal to 1 
        dft_projected = torch.fft.rfft(torch.cat([filler_embs[1:], role_embs], dim=0), dim=-1).abs()
        np.testing.assert_almost_equal(dft_projected.mean().item(), 1, decimal=5)
        np.testing.assert_almost_equal(dft_projected.min().item(), 1, decimal=5)
        np.testing.assert_almost_equal(dft_projected.max().item(), 1, decimal=5)
        np.testing.assert_allclose(dft_projected, torch.ones_like(dft_projected), atol=1e-4, rtol=1e-5)

class VSAConverterConvertSTree(absltest.TestCase): 
    def test_single_node_tree(self): 
        n_fillers = 2
        hypervec_dim = 7
        filler_weights = torch.Tensor([[0, 0, 0, 0, 0, 0, 0],
                                       [1, 2, 4, 8, 16, 32, 64],
                                       [2, 4, 8, 16, 32, 64, 128]])
        role_weights = torch.randn_like(filler_weights)
        tree = torch.Tensor([2])
        
        vsa_converter = get_vsa_instance(n_fillers, hypervec_dim, VSATypes.HRR_STANDARD, 
                               bind_root=False, filler_weights=filler_weights, role_weights=role_weights)
        hrr_rep = vsa_converter(tree.unsqueeze(0)) 
        np.testing.assert_allclose(hrr_rep.squeeze(0).numpy(), np.array([2, 4, 8, 16, 32, 64, 128]))
    
    def test_single_node_tree_root_role(self): 
        n_fillers = 2
        hypervec_dim = 7
        filler_weights = torch.Tensor([[0, 0, 0, 0, 0, 0, 0],
                                       [1, 2, 4, 8, 16, 32, 64],
                                       [2, 4, 8, 16, 32, 64, 128]])
        role_weights = torch.randn(size=(2, hypervec_dim))
        role_weights = torch.concat([role_weights, 
                                     torch.Tensor([1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7]).unsqueeze(0)], dim=0)
        # root role has position 3 in the role_weights dict 
        # 
        tree = torch.Tensor([2])
        vsa_converter = get_vsa_instance(n_fillers, hypervec_dim, VSATypes.HRR_STANDARD, 
                               bind_root=True, filler_weights=filler_weights, role_weights=role_weights)
        hrr_rep = vsa_converter.encode_stree(tree.unsqueeze(0)) 
        # expected result is circ conv between filler_weights[1] and role_weights[positions.role_index]
        expected_result = (254/7)*torch.ones_like(filler_weights[0])
        np.testing.assert_allclose(hrr_rep.squeeze(0).numpy(), expected_result.numpy(), atol=1e-8, rtol=1e-6)
        
    def test_multinode_tree_balanced(self): 
        n_fillers = 3
        hypervec_dim = 4
        filler_weights = torch.Tensor([[0, 0, 0, 0],
                                       [1, 1, 1, 1], 
                                       [2, 3, 4, 5], 
                                       [1, 2, 3, 4]])
        role_weights = torch.Tensor([[1, 9, 8, 3],
                                     [2, 4, 8, 16]])
        tree = torch.Tensor([2, 3, 1])
        #       [2, 3, 4, 5]
        # [1, 2, 3, 4] [1, 1, 1, 1]
        vsa_converter = get_vsa_instance(n_fillers, hypervec_dim, VSATypes.HRR_STANDARD, 
                               bind_root=False, filler_weights=filler_weights, role_weights=role_weights)
        hrr_rep = vsa_converter.encode_stree(tree.unsqueeze(0))
        # expected result
        left_part = torch.Tensor([67, 52, 41, 50])
        right_part = torch.Tensor([30, 30, 30, 30])
        expected_result = left_part + right_part + filler_weights[2]
        np.testing.assert_allclose(hrr_rep.squeeze(0).numpy(), expected_result.numpy(), atol=1e-8, rtol=1e-6)
    
    def test_multinode_tree_balanced(self): 
        n_fillers = 3
        hypervec_dim = 4
        filler_weights = torch.Tensor([[0, 0, 0, 0],
                                       [1, 2, 1, 1], 
                                       [2, 3, 4, 5], 
                                       [1, 2, 3, 4]])
        role_weights = torch.Tensor([[1, 9, 8, 3],
                                     [2, 4, 8, 16]])
        tree = torch.Tensor([2, 3, 1, 1, 1, 2, 3])
        #       [2, 3, 4, 5]
        # [1, 2, 3, 4] [1, 1, 1, 1]
        # [ 1 1]        [ 2 3]
        vsa_converter = get_vsa_instance(n_fillers, hypervec_dim, VSATypes.HRR_STANDARD, 
                               bind_root=False, filler_weights=filler_weights, role_weights=role_weights)
        hrr_rep = vsa_converter.encode_stree(tree.unsqueeze(0))
        # expected result
        children_left = torch.Tensor([24, 22, 30, 29]) + torch.Tensor([46, 32, 34, 38])
        subtree_at_node_3 = filler_weights[3] + children_left  # (71, 56, 67, 71)
        children_right = torch.Tensor([88, 73, 62, 71]) + torch.Tensor([74, 88, 86, 52])
        subtree_at_node_1 = children_right + filler_weights[1] # (163, 163, 149, 124)
        left_part = torch.Tensor([1414, 1464, 1352, 1335]) 
        right_part = torch.Tensor([4622, 4354, 4238, 4756])
        expected_result = left_part + right_part + filler_weights[2]
        np.testing.assert_allclose(hrr_rep.squeeze(0).numpy(), expected_result.numpy(), atol=1e-8, rtol=1e-6)
    
    
    def test_multinode_tree_balanced_root_role(self): 
        n_fillers = 3
        hypervec_dim = 4
        filler_weights = torch.Tensor([[0, 0, 0, 0], 
                                       [1, 1, 1, 1], 
                                       [2, 3, 4, 5], 
                                       [1, 2, 3, 4]])
        role_weights = torch.Tensor([[1, 9, 8, 3],
                                     [2, 4, 8, 16],
                                     [1/2, 1/3, 1/4, 1/2]])
        tree = torch.Tensor([2, 3, 1])
        #       [2, 3, 4, 5]
        # [1, 2, 3, 4] [1, 1, 1, 1]
        vsa_converter = get_vsa_instance(n_fillers, hypervec_dim, VSATypes.HRR_STANDARD, 
                               bind_root=True, filler_weights=filler_weights, role_weights=role_weights)
        hrr_rep = vsa_converter(tree.unsqueeze(0))
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
            [0, 0, 0],
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
        vsa_converter = get_vsa_instance(n_fillers, hypervec_dim, VSATypes.HRR_STANDARD, 
                               bind_root=False, filler_weights=filler_weights, role_weights=role_weights)
        # get hrr rep of tree rooted at 4
        ll_tree = torch.Tensor([14, 21, 9]) + torch.Tensor([17/6, 13/4, 31/12])
        
        # get hrr rep of tree rooted at 3
        l_tree = torch.Tensor([2, 2, 1]) + torch.Tensor([2641/144, 2971/144, 2604/144])
        
        # now get the hrr rep of the entire tree 
        left_part = torch.Tensor([38343/1728, 39514/1728, 38311/1728])
        right_part = torch.Tensor([13/6, 3, -17/6]) 
        expected_result = torch.Tensor([1, 2, 3]) + left_part + right_part
        
        hrr_rep = vsa_converter(tree.unsqueeze(0))
        np.testing.assert_allclose(hrr_rep.squeeze(0).numpy(), expected_result.numpy(), atol=1e-8, rtol=1e-6)
        
    
    def test_multinode_tree_unbalanced_root_role(self): 
        n_fillers = 4
        hypervec_dim = 2
        filler_weights = torch.Tensor([
            [0, 0],
            [1, 2], 
            [2, 8],
            [-3, 7],
            [0, 1]
        ])
        role_weights = torch.Tensor([
            [1/2, -1/2],
            [1/3, 4/5],
            [1, 9]
        ])
        tree = torch.Tensor([2, 4, 1, 0, 1, 3, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        #          2
        #    4           1
        #  0   1       3    0
        # 0  0  0  0  1  0  0  0   
        vsa_converter = get_vsa_instance(n_fillers, hypervec_dim, VSATypes.HRR_STANDARD, 
                               bind_root=True, filler_weights=filler_weights, role_weights=role_weights)
        # find rep of tree rooted at 3
        # [-3, 7] + l \ast [1, 2]
        rl_tree = torch.Tensor([-3, 7]) + torch.Tensor([-1/2, 1/2])
        
        # find rep of tree rooted at 4 
        # [0, 1] + r \ast [1, 2]
        l_tree = torch.Tensor([0, 1]) + torch.Tensor([29/15, 22/15])
        
        # find rep of tree rooted at 1
        # [1, 2] + l \ast lr_tree 
        r_tree = torch.Tensor([1, 2]) + torch.Tensor([-11/2, 11/2])
        
        # find rep of tree rooted at root
        # [2, 8] \ast root 
        root_rep = torch.Tensor([74, 26])
        # expected = root_rep + l \ast l_tree + r \ast r_tree
        l_part = torch.Tensor([-4/15, 4/15])
        r_part = torch.Tensor([9/2, -11/10])
        expected_result = root_rep + l_part + r_part

        hrr_rep = vsa_converter(tree.unsqueeze(0))
        np.testing.assert_allclose(hrr_rep.squeeze(0).numpy(), expected_result.numpy(), atol=1e-8, rtol=1e-6)

class VSAConverterDecodeVSA(absltest.TestCase): 
    """
    def test_decode_single_node_tree_root_unbound(self):
        n_fillers = 3
        hypervec_dim = 3
        filler_weights = torch.Tensor([[1/math.sqrt(3), 1/math.sqrt(3), 1/math.sqrt(3)],
                                       [1/math.sqrt(14), 2/math.sqrt(14), -3/math.sqrt(14)],
                                       [-5/math.sqrt(42), 4/math.sqrt(42), 1/math.sqrt(42)]])
        role_weights = torch.randn_like(filler_weights)
        tree = torch.Tensor([2]) 
        
        vsa_converter = get_vsa_instance(n_fillers, hypervec_dim, VSATypes.HRR_STANDARD, 
                               bind_root=False, filler_weights=filler_weights, role_weights=role_weights, 
                               max_d=1)
        hrr_rep = vsa_converter(tree.unsqueeze(0)) 
        # precondition
        np.testing.assert_allclose(hrr_rep.squeeze(0).numpy(), np.array([-5/math.sqrt(42), 4/math.sqrt(42), 1/math.sqrt(42)]))
        
        decoded = vsa_converter.decode_vsymbolic(vsa=hrr_rep)
        np.testing.assert_allclose(torch.Tensor([0, 0, 1]).reshape(1, 1, 3).numpy(), decoded, atol=1e-6, rtol=1e-6)
    
    
    def test_decode_single_node_tree_root_bound(self): 
        n_fillers = 3
        hypervec_dim = 3
        filler_weights = torch.Tensor([[1/math.sqrt(3), 1/math.sqrt(3), 1/math.sqrt(3)],
                                       [1/math.sqrt(14), 2/math.sqrt(14), -3/math.sqrt(14)],
                                       [-5/math.sqrt(42), 4/math.sqrt(42), 1/math.sqrt(42)]])
        role_weights = torch.randn_like(filler_weights)
        role_weights = torch.cat([role_weights[:-1], torch.Tensor([-1/2, 1/4, 1/5]).unsqueeze(0)], dim=0)
        tree = torch.Tensor([2]) 
        
        vsa_converter = get_vsa_instance(n_fillers, hypervec_dim, VSATypes.HRR_STANDARD, 
                               bind_root=True, filler_weights=filler_weights, role_weights=role_weights,
                               max_d=1)
        hrr_rep = vsa_converter(tree.unsqueeze(0)) 
        # precondition
        np.testing.assert_allclose(hrr_rep.squeeze(0).numpy(), (1/math.sqrt(42))*np.array([71/20, -61/20, -1/2]), atol=1e-7, rtol=1e-6)
        
        decoded = vsa_converter.decode_vsymbolic(vsa=hrr_rep)
        np.testing.assert_allclose(torch.Tensor([0, 0, 1]).reshape(1, 1, 3).numpy(), decoded, atol=1e-6, rtol=1e-6)
    
    def test_decode_multinode_tree_full_root_unbound(self): 
        n_fillers = 4
        hypervec_dim = 1024
        d = 2
        tree = torch.Tensor([1, 2, 3])
        
        vsa_converter = get_vsa_instance(n_fillers, hypervec_dim, VSATypes.HRR_STANDARD, 
                                         bind_root=False, filler_weights=None, 
                                         role_weights=None, max_d=d)

        hrr_rep = vsa_converter(tree.unsqueeze(0))

        # output has shape (1, n_nodes, N_{F})
        sims = vsa_converter.decode_vsymbolic(hrr_rep)
        idxs = torch.argmax(sims, dim=-1)
        np.testing.assert_array_equal(tree.unsqueeze(0), idxs)
    """
    
    def test_decode(self): 
        n_fillers = 4
        hypervec_dim = 24
        tree = torch.Tensor([2, 1, 1, 3, 1, 2, 2])
        #       [2, 3, 4, 5]
        # [1, 2, 3, 4] [1, 1, 1, 1]
        # [ 1 1]        [ 2 3]
        vsa_converter = get_vsa_instance(n_fillers, hypervec_dim, VSATypes.HRR_NON_COMMUTATIVE, 
                               bind_root=False)
        
        fillers = vsa_converter.filler_emb.weight 
        roles = vsa_converter.role_emb.weight 
        
        """
        Fillers are Parameter containing:
        tensor([[ 0.0000,  0.0000,  0.0000,  0.0000],
                [-0.4911,  0.5941,  0.4911,  0.4059],
                [-0.3289, -0.4698, -0.6711,  0.4698],
                [-0.9577, -0.2012, -0.0423,  0.2012]])
        Roles are Parameter containing:
        tensor([[-0.3214,  0.4670, -0.6786, -0.4670],
                [ 0.4525, -0.2872, -0.4525, -0.7128]])
        """
        
        # preconditions that projections unitary
        assert_complex_projection_unitary(fillers[1:])
        assert_complex_projection_unitary(roles)
        
        hrr_rep = vsa_converter.encode_stree(tree.unsqueeze(0))
        
        # expected result
        
        #np.testing.assert_allclose(hrr_rep.squeeze(0).numpy(), expected_result.numpy(), atol=1e-8, rtol=1e-6)

        # determine which node is root 
        root_idx_sims = torch.cosine_similarity(hrr_rep.unsqueeze(0), vsa_converter.filler_emb.weight.unsqueeze(0), dim=-1)
        root_idx = torch.argmax(root_idx_sims, dim=-1)
        np.testing.assert_equal(root_idx, tree[0])
        print(f'Root idx is {torch.argmax(root_idx_sims)}, with sims {root_idx_sims}')
        
        # cosine sim
        unbound_left = vsa_converter.vsa_operator.unbind_op(vsa_converter.left_role, hrr_rep)
        unbound_left_sims = torch.cosine_similarity(unbound_left.unsqueeze(0), vsa_converter.filler_emb.weight.unsqueeze(0), dim=-1)
        unbound_left_idx = torch.argmax(unbound_left_sims, dim=-1)
        print(f'Unbound left is {unbound_left} with sims {unbound_left_sims} w idx {unbound_left_idx}')
        
        unbound_right = vsa_converter.vsa_operator.unbind_op(vsa_converter.right_role, hrr_rep)
        unbound_right_sims = torch.cosine_similarity(unbound_right.unsqueeze(0), vsa_converter.filler_emb.weight.unsqueeze(0), dim=-1)
        unbound_right_idx = torch.argmax(unbound_right_sims, dim=-1)
        print(f'Unbound right is {unbound_right} with sims {unbound_right_sims} w idx {unbound_right_idx}')
        
        unbound_ll = vsa_converter.vsa_operator.unbind_op(vsa_converter.left_role, unbound_left)
        unbound_ll_sims = torch.cosine_similarity(unbound_ll.unsqueeze(0), vsa_converter.filler_emb.weight.unsqueeze(0), dim=-1)
        unbound_ll_idx = torch.argmax(unbound_ll_sims, dim=-1)
        print(f'Unbound ll is {unbound_ll} with sims {unbound_ll_sims} w idx {unbound_ll_idx}')
        
        unbound_lr = vsa_converter.vsa_operator.unbind_op(vsa_converter.right_role, unbound_left)
        unbound_lr_sims = torch.cosine_similarity(unbound_lr.unsqueeze(0), vsa_converter.filler_emb.weight.unsqueeze(0), dim=-1)
        unbound_lr_idx = torch.argmax(unbound_lr_sims, dim=-1)
        print(f'Unbound lr is {unbound_lr} with sims {unbound_lr_sims} w idx {unbound_lr_idx}')
        
        unbound_rl = vsa_converter.vsa_operator.unbind_op(vsa_converter.left_role, unbound_right)
        unbound_rl_sims = torch.cosine_similarity(unbound_rl.unsqueeze(0), vsa_converter.filler_emb.weight.unsqueeze(0), dim=-1)
        unbound_rl_idx = torch.argmax(unbound_rl_sims, dim=-1)
        print(f'Unbound rl is {unbound_rl} with sims {unbound_rl_sims} w idx {unbound_rl_idx}')
        
        unbound_rr = vsa_converter.vsa_operator.unbind_op(vsa_converter.right_role, unbound_right)
        unbound_rr_sims = torch.cosine_similarity(unbound_rr.unsqueeze(0), vsa_converter.filler_emb.weight.unsqueeze(0), dim=-1)
        unbound_rr_idx = torch.argmax(unbound_rr_sims, dim=-1)
        print(f'Unbound rr is {unbound_rr} with sims {unbound_rr_sims} w idx {unbound_rr_idx}')
        
        #left_idx_sims = torch.cosine_similarity(unbound_left.unsqueeze(0), vsa_converter.inv_left_role)
        #left_idx = torch.argmax(left_idx_sims)
        #print(f'Left idx is {left_idx}, with sims {left_idx_sims}')
        # maybe it's worth trying to remove the unbound roles -> 
        # also provide the unbound roles and the confidences into a network that can also be trained???
        # for later on?? 
        
    
    def test_decode_multinode_tree_full_root_unbound(self): 
        n_fillers = 4
        hypervec_dim = 32
        d = 3
        tree = torch.Tensor([1, 2, 3, 2, 3, 1, 2])
        
        vsa_converter = get_vsa_instance(n_fillers, hypervec_dim, VSATypes.HRR_NON_COMMUTATIVE, 
                                         bind_root=False, filler_weights=None, 
                                         role_weights=None, max_d=d)

        hrr_rep = vsa_converter(tree.unsqueeze(0))

        # output has shape (1, n_nodes, N_{F})
        sims = vsa_converter.decode_vsymbolic(hrr_rep)
        idxs = torch.argmax(sims, dim=-1)
        np.testing.assert_array_equal(tree.unsqueeze(0), idxs)


    """
    def decode_multinode_tree_full_root_bound(self): 
        pass
    
    def decode_multinode_tree_unbalanced_root_unbound(self): 
        pass 
    
    def decode_multinode_tree_unbalanced_root_bound(self): 
        pass 
    
    def decode_multinode_tree_noisy(self): 
        pass 

#class VSAUtilsTestBatchHRR(absltest.TestCase): 
#    def test_single_elem_batch_no_empty_nodes(self): 
#        pass 
#    
#    def test_single_elem_batch_empty_nodes(self): 
#        pass
#    
#    def test_batched_no_empty_nodes(self): 
#        pass 
#    
#    def test_batched_some_empty_nodes(self): 
#        pass 
"""

# continue doing this when home

def assert_complex_projection_unitary(x: torch.Tensor) -> None:
    dft_projected = torch.fft.rfft(x, dim=-1).abs()
    np.testing.assert_almost_equal(dft_projected.mean().item(), 1, decimal=5)
    np.testing.assert_almost_equal(dft_projected.min().item(), 1, decimal=5)
    np.testing.assert_almost_equal(dft_projected.max().item(), 1, decimal=5)
    np.testing.assert_allclose(dft_projected, torch.ones_like(dft_projected), atol=1e-4, rtol=1e-5)

    
if __name__ == '__main__': 
    absltest.main()
    
    
    
    