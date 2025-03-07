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
from hrr_ops import complexMagProj

class TestHRROps(absltest.TestCase): 
    def test_complex_mag_proj(self): 
        x = torch.Tensor([1, 1.5, 2.5, -1])
        projected = complexMagProj(x) 
        expected = torch.Tensor([0.234, 0.4285, 0.757, -0.4285])
        np.testing.assert_array_almost_equal(projected, expected)
    
    def test_complex_mag_proj_batched(self): 
        pass 
    
    def test_circular_conv(self):
        pass 
    
    def test_circular_corr(self): 
        pass 
    
    def test_get_approx_inv_non_unitary(self): 
        pass 
    
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