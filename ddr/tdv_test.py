import os,sys

_lib = os.path.join( os.path.dirname(os.path.abspath(__file__)),"../")
if _lib not in sys.path:
    sys.path.insert(0, _lib)

import torch
import numpy as np

from parameterized import parameterized, parameterized_class
import unittest

from ddr.tdv import TDV, ActivationFactory

# to run execute: python -m unittest [-v] ddr.tdv

R_types = {
    'tdv': TDV,
}

class GradientTest(unittest.TestCase):
    
   @parameterized.expand([
       ("tdv",              "tanh",       "tanh",      "student-t", False, True, True  ),
       ("tdv",              "tanh",       "tanh",      "student-t", False, True,       ),
       ("tdv",              "student-t",  "student-t", "linear",    True,  True,       ),
       ("tdv",              "tanh",       "student-t", "linear",    False, True,       ),
       ("tdv",              "student-t",  "student-t", "linear"                ,       ),
       ("tdv",              "tanh",       "student-t", "linear"                ,       ),
       ("tdv",              "tanh",       "student-t", "linear"                ,       ),
   ])   
   def test_tdv_gradient(self,name, act, act_fin, psi, bias=False, embedding=None, lambda_mul=None):
        if embedding:
            print(f"testing: {name}, [{act},{act_fin},{psi}] with embedding")
        else:
            print(f"testing: {name}, [{act},{act_fin},{psi}]")
        # setup the data
        ch = 3
        x = np.random.rand(2,ch,16,32)
        N,C,H,W = x.shape
        x = torch.from_numpy(x).cuda()

        # define the TDV regularizer
        config ={
            'in_channels': ch,
            'out_channels': 1,
            'num_features': 4,
            'num_scales': 3,
            'num_mb': 2,
            'multiplier': 2,
            'act' : act,
            'act_fin':act_fin,
            'psi': psi,
            'bias_w1':False,
        }
        lambda_mul
        if lambda_mul is not None:
            lambda_mul = torch.randn([N,1,H,W], device=x.device,dtype=x.dtype)

        if embedding:
            embedding = []
            f = config['num_features']
            cf, cH,cW = f,H,W
            for i in range(config['num_scales']):
                embedding.append(
                    torch.randn([N,cf,cH,cW], device=x.device,dtype=x.dtype)
                )
                cf,cH,cW = cf*2, cH//2, cW//2
        else:
            embedding=None

        R_class = R_types[name]
        R = R_class(config).double().cuda()
        R.conf_pos = 0

        def compute_loss(scale):
            if embedding:
                return torch.sum(R.energy(scale*x, embedding=embedding, lambda_mul=lambda_mul))
            else:
                return torch.sum(R.energy(scale*x, lambda_mul=lambda_mul))
        
        scale = 1.
        
        # compute the gradient using the implementation
        if embedding:
            grad_scale = torch.sum(x*(R.grad(scale*x, embedding=embedding, lambda_mul=lambda_mul)[1])).item()
        else:
            grad_scale = torch.sum(x*(R.grad(scale*x, lambda_mul=lambda_mul)[1])).item()

        # check it numerically
        epsilon = 1e-4
        with torch.no_grad():
            l_p = compute_loss(scale+epsilon).item()
            l_n = compute_loss(scale-epsilon).item()
            grad_scale_num = (l_p - l_n) / (2 * epsilon)

        condition = np.abs(grad_scale - grad_scale_num) < 1e-3
        msg=f'   grad_scale: {grad_scale:.7f} num_grad_scale {grad_scale_num:.7f} success: {condition} , {config}'
        print(msg)
        self.assertTrue(condition, msg)

    
   @parameterized.expand([
       ( "softplus" ,),
       ( "tanh" ,),
       ( "student-t" ,),
   ])    
   def test_activation_gradient(self, name ): 
        print(f"testing activation function: {name}")
        # setup the data
        x = np.random.rand(1024)
        x = torch.from_numpy(x).cuda()

        act = ActivationFactory( name).double().cuda() 

        def compute_loss(scale, id=None):
            a0, a1 = act.forward(scale*x)
            if id is None:
                return a1.sum() + a1.sum()
            elif id == 0:
                return a0.sum()
            elif id==1: 
                return a1.sum()
            else:
                raise RuntimeError("wrong id")

        scale = 1.
        
        x_param = torch.nn.Parameter(x, requires_grad=True)
        act_x, act_dx = act.forward(scale*x_param)


        # compute the gradient using the implementation
        # d( phi(a*x) )/da = x * phi'(a*x)
        loss0 = act_x.sum() 
        loss0.backward(retain_graph=True)
        grad0_scale = (x * x_param.grad).sum().item()
        x_param.grad.zero_()
        loss1 = act_dx.sum()
        loss1.backward(retain_graph=True)
        grad1_scale = (x * x_param.grad).sum().item()
        # x_grad_ana  = x_param.grad

        # check it numerically
        epsilon = 1e-4
        with torch.no_grad():
            l_p = compute_loss(scale+epsilon,0).item()
            l_n = compute_loss(scale-epsilon,0).item()
            grad0_scale_num = (l_p - l_n) / (2 * epsilon)
            l_p = compute_loss(scale+epsilon,1).item()
            l_n = compute_loss(scale-epsilon,1).item()
            grad1_scale_num = (l_p - l_n) / (2 * epsilon)

        r_delta0 = np.abs(grad0_scale - grad0_scale_num) / np.clip(np.abs(grad0_scale)+ np.abs(grad0_scale_num), 1e-9, np.inf)
        r_delta1 = np.abs(grad1_scale - grad1_scale_num) / np.clip(np.abs(grad0_scale)+ np.abs(grad0_scale_num), 1e-9, np.inf)
        cond0 = r_delta0 < 1e-3
        cond1 = r_delta1 < 1e-3
        print(f'   grad_scale: {grad0_scale:.7f} num_grad_scale {grad0_scale_num:.7f} success: {cond0}')
        print(f'   grad_scale: {grad1_scale:.7f} num_grad_scale {grad1_scale_num:.7f} success: {cond1}')
        self.assertTrue(cond0 and cond1)



# to run execute: python -m unittest [-v] ddr.tdv_test
if __name__ == "__main__":
    t = GradientTest()
    # t.test_activation_gradient("tanh")
    # t.test_activation_gradient_0_softplus()
    # t.test_activation_gradient_1_tanh()
    unittest.main()

