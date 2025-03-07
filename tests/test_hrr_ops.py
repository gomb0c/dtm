# to do
# test initialisation of seed vecs
# compare with HRR paper - ensure complex unit magnitude 

# then test circ conv
# test circ corr
# test inverse op
# test conv of inverse is equal to circ corr given that vecs have complex unit magnitude

import math
import numpy as np
import torch
from absl.testing import absltest 
from hrr_ops import complexMagProj, circular_conv, circular_corr

class TestHRROps(absltest.TestCase): 
    def test_complex_mag_proj(self): 
        x = torch.Tensor([1, 1.5, 2.5, -1])
        projected = complexMagProj(x) 
        expected = torch.Tensor([0.2427521, 0.4287465, 0.7572479, -0.4287465])
        np.testing.assert_allclose(projected, expected)
    
    def test_complex_mag_proj_batched(self): 
        x = torch.Tensor([[-3, 4/3, 7, 10],
                          [2, 8, 1.5, -2]])
        projected = complexMagProj(x) 
        expected = torch.Tensor([[-0.3778440, 0.1725340, 0.3778440, 0.8274650],
                                 [0.0249688,0.9993762,-0.0249688,0.0006238]])
        np.testing.assert_allclose(projected, expected, atol=1e-5,
                                             rtol=1e-6)
    
    def test_circular_conv(self):
        a = torch.zeros(size=(5, 3))
        b = torch.randn_like(a)
        np.testing.assert_allclose(circular_conv(a, b), torch.zeros_like(a))

        a = torch.Tensor([[2/3, -2/3, 1/3],
                          [1/5, 1/8, -8/9]])
        b = torch.Tensor([[10, -11, 23],
                          [1/10, 2/10, -1/50]])
        expected = torch.Tensor([[-12.3333333,-6.3333333,26.0000000],
                                 [-0.1602778,0.0702778, -0.0678889]])
        actual = circular_conv(a, b)
        np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=1e-6)
        
    def test_circular_corr(self): 
        a = torch.zeros(size=(10, 2))
        b = torch.randn_like(a) 
        np.testing.assert_allclose(circular_corr(a,b), torch.zeros_like(a))
        
        a = torch.Tensor([[1/4, -2/9, 3/31, 9],
                          [-1/5, 1/8, 1/10, -11]])
        b = torch.Tensor([[-11.1, 23.3, 12, -0.11],
                         [-0.5, 2.4, -3.1, 4.2]])
        expected = torch.Tensor([[-7.781488,-96.752309,211.650250,112.693000], 
                                 [-46.11,5.0525,-25.305,33.437500]])
        actual = circular_corr(a, b)
        np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=1e-6)
        
    def test_circular_corr_is_inverse(self): 
        ''' Precondition for circular correlation to unbind is that the vector we want to unbind
        must have complex magnitude equal to 1'''
        a = complexMagProj(torch.Tensor([2/3, -2/3, 1/3]).unsqueeze(0))
        b = torch.Tensor([10, -11, 23]).unsqueeze(0)
        conv = circular_conv(a, b)
        
        # precondition
        np.testing.assert_allclose(a, torch.Tensor([(math.sqrt(13) + 5)/(3*math.sqrt(13)), 
                                                    (math.sqrt(13) - 7)/(3*math.sqrt(13)),
                                                    (math.sqrt(13) + 2) / (3*math.sqrt(13))]).unsqueeze(0), 
                                   atol=1e-7, rtol=1e-6)
        
        expected_conv = 1/(3*math.sqrt(13))*torch.Tensor([22*math.sqrt(13) - 133, 
                                                        22*math.sqrt(13) - 79, 
                                                        22*math.sqrt(13) + 212]).unsqueeze(0)
        np.testing.assert_allclose(conv, expected_conv, atol=1e-5, rtol=1e-6)
        
        # test
        hat_b = circular_corr(a, conv)
        np.testing.assert_allclose(hat_b, b, atol=1e-8, rtol=1e-6)
        
        c = torch.randn_like(b)
        conv = circular_conv(a, c)
        hat_c = circular_corr(a, conv)
        np.testing.assert_allclose(hat_c, c, atol=1e-8, rtol=1e-6)
    
    def test_get_pseudo_inv_non_unitary(self): 
        a = torch.Tensor([])
    
    def test_get_inv_non_unitary(self): 
        pass 
    
    def test_get_inv_unitary(self): 
        # test inverse has correct complex magnitude
        # test equivalent to pseudo-inverse
        pass 
    
    def test_recoverability(self): 
        pass 
    
    
    
    
if __name__ == '__main__': 
    absltest.main()