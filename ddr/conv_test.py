import os,sys

_lib = os.path.join( os.path.dirname(os.path.abspath(__file__)),"../")
if _lib not in sys.path:
    sys.path.insert(0, _lib)

import torch
import numpy as np

from parameterized import parameterized, parameterized_class
import unittest

from ddr.conv import Conv2d, ConvScale2d, ConvScaleTranspose2d

# to run execute: python -m unittest [-v] ddr.tdv

conv_types = {
    'conv2d': Conv2d,
    'conv2dS': ConvScale2d,
    'conv2dST': ConvScaleTranspose2d,
}

class GradientTest(unittest.TestCase):
    
   @parameterized.expand([
       # Op,      Bias
       ("conv2dST", True,),
       ("conv2dST", False,),
       ("conv2d", True,),
       ("conv2d", False,),
       ("conv2dS", True,),
       ("conv2dS", False,),
   ])    
   def test_conv2d_gradient(self,name, bias):
        print(f"testing: {name}, bias:{bias}")
        # Concept of this gradient test:
        # K ( a * x + b) => x @ K^T (a *x + b)

        # setup the data
        ch_in = 9
        ch_out = 1
        x = np.random.rand(2,ch_in,32,64)
        x = torch.from_numpy(x).cuda()
        y = torch.ones([2,ch_out,32,64],dtype=x.dtype).cuda()
        # y = np.random.rand(2,ch_out,32,64)
        # y = torch.from_numpy(y).cuda()
        if name == "conv2dST":
            tmp = ch_out
            ch_out = ch_in
            ch_in = tmp
            def compute_loss(scale):
                return torch.sum(conv.forward(x*scale, output_shape=y.shape))
        else:
            def compute_loss(scale):
                return torch.sum(conv.forward(x*scale))

        conv_class = conv_types[name]
        conv = conv_class(
                in_channels=ch_in,
                out_channels=ch_out,
                kernel_size=3,
                invariant=False,
                stride=1,  
                bias=bias, 
                zero_mean=False,
                bound_norm=False
        ).double().cuda() # Gradient Test => double

        
        scale = 1.
        
        # compute the gradient using the implementation
        # y = 1 @ K ( a * x + b)
        #  => dy/da = x @ K^T @ 1
        grad_scale = torch.sum(x*conv.backward(y)).item()

        # check it numerically
        epsilon = 1e-4
        with torch.no_grad():
            l_p = compute_loss(scale+epsilon).item()
            l_n = compute_loss(scale-epsilon).item()
            grad_scale_num = (l_p - l_n) / (2 * epsilon)

        condition = np.abs(grad_scale - grad_scale_num) < 1e-7
        print(f'   grad_scale: {grad_scale:.7f} num_grad_scale {grad_scale_num:.7f} success: {condition}')
        self.assertTrue(condition)

    
    

# to run execute: python -m unittest [-v] ddr.tdv_test
if __name__ == "__main__":
    unittest.main()

