
import torch
import numpy as np

from .regularizer import Regularizer
from .conv import *

import unittest

from typing import Optional, List

from common_utils.torch_script_logging import TORCHSCRIPT_EXPORT

__all__ = ['TDV','MaskPredictor','ActivationFactory']

class SoftPlus_fun2(torch.autograd.Function):
    """returns Soft-Plus as well as its derivative in the forward path"""
    @staticmethod
    def forward(ctx, x):
        soft_x = torch.nn.functional.softplus(x) # Numerically stable verison     torch.log( 1+ torch.exp(x) )
        grad_soft_x_dx = torch.nn.functional.sigmoid(x)
        ctx.save_for_backward(grad_soft_x_dx)
        return soft_x, grad_soft_x_dx

    @staticmethod
    def backward(ctx, grad_fx, grad_dfx):
        sigmoid_x = ctx.saved_tensors[0]
        dfx  = sigmoid_x  # 1st derivative 
        d2fx = dfx * (1-dfx)                   # 2nd derivative
        return grad_fx * dfx + grad_dfx * d2fx

class SoftPlus2(torch.nn.Module):
    """returns Softplus and its derivative in the forward path"""
    def __init__(self):
        super(SoftPlus2, self).__init__()
    def forward(self, x):
        """returns ln(1+exp(x), sigmoid(x) in the forward path
            softplus2_mod = SoftPlus2()
            sp_x,sp_x_grad = softplus2(x)
        """
        return SoftPlus_fun2().apply(x)


class Tanh_fun2(torch.autograd.Function):
    """returns Tanh and its derivative in the forward path"""
    @staticmethod
    def forward(ctx, x):
        tanhx = torch.tanh(x)
        ctx.save_for_backward(tanhx)
        fx = tanhx        # forward function
        dfx = 1-tanhx**2  # 1st derivative 
        return fx, dfx

    @staticmethod
    def backward(ctx, grad_fx, grad_dfx):
        tanhx = ctx.saved_tensors[0]
        tanhx_sq = tanhx**2        
        dfx = 1-tanhx_sq          # 1st derivative 
        d2fx = -(dfx) * 2 * tanhx # 2nd derivative
        return grad_fx * dfx + grad_dfx * d2fx

class Tanh2(torch.nn.Module):
    """returns Tanh and its derivative in the forward path"""
    def __init__(self):
        super(Tanh2, self).__init__()
    def forward(self, x):
        """returns Tanh(x) and d/dx(tanh(x)) in the forward path
            tanh2_mod = Tanh2()
            tanhx,tanhx_grad = tanh2_mod(x)
        """
        return Tanh_fun2().apply(x)


class Tanh2Inference(torch.nn.Module):
    """returns Tanh and its derivative in the forward path"""
    def __init__(self):
        super(Tanh2Inference, self).__init__()
    def forward(self, x):
        """returns Tanh(x) and d/dx(tanh(x)) in the forward path
            tanh2_mod = Tanh2()
            tanhx,tanhx_grad = tanh2_mod(x)
        """
        tanhx = torch.tanh(x)
        fx = tanhx        # forward function
        dfx = 1-tanhx**2  # 1st derivative
        return fx, dfx


class StudentT_fun2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        d = 1+alpha*x**2
        return torch.log(d)/(2*alpha), x/d

    @staticmethod
    def backward(ctx, grad_in1, grad_in2):
        x = ctx.saved_tensors[0]
        d = 1+ctx.alpha*x**2
        return (x/d) * grad_in1 + (1-ctx.alpha*x**2)/d**2 * grad_in2, None


class StudentT2(torch.nn.Module):
    def __init__(self,alpha):
        super(StudentT2, self).__init__()
        self.alpha = alpha
    def forward(self, x):
        return StudentT_fun2().apply(x, self.alpha)

def ActivationFactory(act_cfg):
    if act_cfg == "student-t":
        act = StudentT2(alpha=1)
    elif act_cfg == "tanh":
        act = Tanh2()
    elif act_cfg == "softplus":
        act = SoftPlus2()
    else:
        raise ValueError(f"wrong config for activation function! {act_cfg}")
    return act

if TORCHSCRIPT_EXPORT:
    def ActivationFactory(act_cfg):
        if act_cfg != 'tanh':
            raise ValueError(f"Torchscript export only implemented for tanh function but found {act_cfg}")
        return Tanh2Inference()

class MicroBlock(torch.nn.Module):

    act_prime: Optional[torch.Tensor] 

    def __init__(self, num_features, bound_norm=False, invariant=False, act="student-t"):
        super(MicroBlock, self).__init__()
        
        self.conv1 = Conv2d(num_features, num_features, kernel_size=3, invariant=invariant, bound_norm=bound_norm, bias=False)
        self.act = ActivationFactory(act)
        self.conv2 = Conv2d(num_features, num_features, kernel_size=3, invariant=invariant, bound_norm=bound_norm, bias=False)

        # save the gradient of the the activation function for the backward path
        self.act_prime = None

    def forward(self, x):
        a, ap = self.act(self.conv1(x))
        self.act_prime = ap
        x = x + self.conv2(a)
        return x

    def backward(self, grad_out):
        act_prime = self.act_prime
        assert act_prime is not None, 'call forward before calling backward!' # required for TorchScript export

        out = grad_out + self.conv1.backward(act_prime*self.conv2.backward(grad_out))
        if not act_prime.requires_grad:
            self.act_prime = None
        return out


class MacroBlock(torch.nn.Module):
    def __init__(self, num_features, num_scales=3, multiplier=1, bound_norm=False, invariant=False, act="student-t", act_fin="student-t"):
        super(MacroBlock, self).__init__()

        self._is_TORCHSCRIPT_EXPORT:bool = TORCHSCRIPT_EXPORT # Hack for Torchscript export

        self.num_scales = num_scales

        # micro blocks
        self.mb = []
        # Final Micro Block (scale=0) has different activation
        self.mb.append(torch.nn.ModuleList([
            MicroBlock(num_features * multiplier**0, bound_norm=bound_norm, invariant=invariant, act=act),
            MicroBlock(num_features * multiplier**0, bound_norm=bound_norm, invariant=invariant, act=act_fin)
            ]) )
        for i in range(1, num_scales-1):
            b = torch.nn.ModuleList([
                MicroBlock(num_features * multiplier**i, bound_norm=bound_norm, invariant=invariant, act=act),
                MicroBlock(num_features * multiplier**i, bound_norm=bound_norm, invariant=invariant, act=act)
            ])
            self.mb.append(b)
        # the coarsest scale has only one microblock
        self.mb.append(torch.nn.ModuleList([
                MicroBlock(num_features * multiplier**(num_scales-1), bound_norm=bound_norm, invariant=invariant, act=act)
        ]))
        self.mb = torch.nn.ModuleList(self.mb)

        # down/up sample
        self.conv_down = []
        self.conv_up = []
        for i in range(1, num_scales):
            self.conv_down.append(
                ConvScale2d(num_features * multiplier**(i-1), num_features * multiplier**i, kernel_size=3, bias=False, invariant=invariant, bound_norm=bound_norm)
            )
            self.conv_up.append(
                ConvScaleTranspose2d(num_features * multiplier**(i-1), num_features * multiplier**i, kernel_size=3, bias=False, invariant=invariant, bound_norm=bound_norm)
            )
        self.conv_down = torch.nn.ModuleList(self.conv_down)
        self.conv_up = torch.nn.ModuleList(self.conv_up)


    def forward(self, x: List[torch.Tensor]):
        assert len(x) == self.num_scales

        # down scale and feature extraction
        # for (i, micro_blocks, conv_down) in zip(range(self.num_scales-1), self.mb, self.conv_down):
        for  i, (micro_blocks, conv_down) in enumerate(zip( self.mb, self.conv_down)):
            # 1st micro block of scale
            x[i] = micro_blocks[0](x[i])
            # down sample for the next scale
            x_i_down = conv_down(x[i])
            x[i+1] = x[i+1] + x_i_down

        # on the coarsest scale we only have one micro block
        x[-1] = self.mb[-1][0](x[-1])

        # up scale the features
        if self._is_TORCHSCRIPT_EXPORT:
            return self._forward_torchscript_helper(x)
        else:
            return self._forward_std(x)

    def _forward_torchscript_helper(self, x:List[torch.Tensor]) -> List[torch.Tensor]:
        """ This is a dirty hack, as TorchScript does not allow processing of ModuleLists with non-literal indices, nor reverse iteration (pytorch 1.8)
            => export unrolled version for a fixed number of iterations

            There might be a fix for this problem in the nightly build
            (see #53410 and #45716 in the PyTroch repo) of PyTroch,
        """
        # TODO: Improve with future Pytorch versions
        if self.num_scales != 3:
            raise ValueError(f"TorchScript Export is currently only implemented for a fixed number of scales num_scales=3 but found {self.num_scales} ")

        # up scale the features
        x_ip1_up = self.conv_up[1](x[2], x[1].shape)
        x[1] = x[1] + x_ip1_up
        x[1] = self.mb[1][1](x[1])

        x_ip1_up = self.conv_up[0](x[1], x[0].shape)
        x[0] = x[0] + x_ip1_up
        x[0] = self.mb[0][1](x[0])
        return x

    @torch.jit.unused
    def _forward_std(self, x:List[torch.Tensor]) -> List[torch.Tensor]:
        for i in range(self.num_scales-1)[::-1]:
           # first upsample the next coarsest scale
           x_ip1_up = self.conv_up[i](x[i+1], x[i].shape)
           # skip connection
           x[i] = x[i] + x_ip1_up
           # 2nd micro block of scale
           x[i] = self.mb[i][1](x[i])
        return x


    def backward(self, grad_x: List[torch.Tensor]):
        # backward of up scale the features
        for i, (micro_blocks, conv_up) in enumerate(zip(self.mb, self.conv_up)):
            # 2nd micro block of scale
            grad_x[i] = micro_blocks[1].backward(grad_x[i])
            # first upsample the next coarsest scale
            grad_x_ip1_up = conv_up.backward(grad_x[i])
            grad_x[i+1] = grad_x[i+1] + grad_x_ip1_up

        # on the coarsest scale we only have one micro block
        grad_x[-1] = self.mb[-1][0].backward(grad_x[-1])

        # down scale and feature extraction
        # up scale the features
        if self._is_TORCHSCRIPT_EXPORT:
            return self._backward_torchscript_helper(grad_x)
        else:
            return self._backward_std(grad_x)

    def _backward_torchscript_helper(self, grad_x:List[torch.Tensor]) -> List[torch.Tensor]:
        # VERY DIRTY HACK - see comment above in forward function.
        # TODO: Improve with future Pytorch versions
        if self.num_scales != 3:
            raise ValueError(f"TorchScript Export is currently only implemented for a fixed number of scales num_scales=3 but found {self.num_scales} ")

        # down scale and feature extraction
        grad_x_down = self.conv_down[1].backward(grad_x[2], grad_x[1].shape)
        grad_x[1] = grad_x[1] + grad_x_down
        grad_x[1] = self.mb[1][0].backward(grad_x[1])

        grad_x_down = self.conv_down[0].backward(grad_x[1], grad_x[0].shape)
        grad_x[0] = grad_x[0] + grad_x_down
        grad_x[0] = self.mb[0][0].backward(grad_x[0])
        return grad_x

    @torch.jit.unused
    def _backward_std(self,grad_x:List[torch.Tensor]) -> List[torch.Tensor]:
        # down scale and feature extraction
        for i in range(self.num_scales-1)[::-1]:
           # down sample for the next scale
           grad_x_i_down = self.conv_down[i].backward(grad_x[i+1], grad_x[i].shape)
           grad_x[i] = grad_x[i] + grad_x_i_down
           # 1st micro block of scale
           grad_x[i] = self.mb[i][0].backward(grad_x[i])
        return grad_x
        

class MaskPredictor(torch.nn.Module):
    """
    A Network that predicts a confidence mask
    """
    def __init__(self, config=None ):
        super(MaskPredictor, self).__init__()
 
        self.in_channels  = config['in_channels']
        self.num_features = config['num_features']
        self.out_channels = config['out_channels']
        self.multiplier = config['multiplier']
        self.num_mb = config['num_mb']
        self.act = config['act']
        self.act_fin_mb = config['act_fin']
        self.num_scales = config['num_scales']
        self.bias_in  = config['bias_in' ] if ('bias_in'  in config) else False
        self.bias_out = config['bias_out'] if ('bias_out' in config) else False
      
        self.no_head = config['no_head'] if ('no_head' in config) else False

        self.embedding_out = config['embedding_out'] if ('embedding_out' in config) else False

        # Construct Network similar to TDV
        self.K1 = Conv2d(self.in_channels, self.num_features, 3, zero_mean=False, invariant=False, bound_norm=False, bias=self.bias_in)

        self.mb = torch.nn.ModuleList([MacroBlock(self.num_features, num_scales=self.num_scales, bound_norm=False,
                                                  invariant=False, multiplier=self.multiplier,  act=self.act, 
                                                  act_fin=self.act_fin_mb if i == self.num_mb-1 else self.act) 
                                        for i in range(self.num_mb)])

        self.KN = None
        if not self.no_head:
            self.KN = Conv2d(self.num_features, self.out_channels, 1, invariant=False, bound_norm=False, bias=self.bias_out) 

        self.Ke = None
        if self.embedding_out:
            self.Ke = torch.nn.ModuleList([MicroBlock(self.num_features * self.multiplier**i, bound_norm=False, invariant=False, act=self.act)
                                                        for i in range(self.num_scales)])


    def forward(self, x, embedding: Optional[List[torch.Tensor]]=None):
        # extract features
        x = self.K1(x)
        # apply mb
        x_list = [x,] + (self.num_scales - 1)*[torch.zeros(1, device=x.device)]
        if embedding is not None:
            # print("use embedding here")
            for s in range(self.num_scales):
                x_list[s] = embedding[s] + x_list[s]

        for macro_block in self.mb:
            x_list = macro_block(x_list)

        # compute the output
        if self.KN is None:
            out = x_list[0]
        else:
            out = self.KN(x_list[0])

        out_emb = None
        if self.Ke is not None:
            #Compute a final transformation of the internal features, so that they can be used by the TDV later
            out_emb = []
            for i_scales in range(self.num_scales):
                out_emb += [ self.Ke[i_scales](x_list[i_scales]) ]

        return out, out_emb

class TDV(Regularizer):
    """
    total deep variation (TDV) regularizer
    """

    @staticmethod
    def potential_linear(x):
        return x

    @staticmethod
    def potential_student_t(x):
        return 0.5*torch.log(1+x**2)

    @staticmethod
    def potential_tanh(x):
        return torch.log(torch.cosh(x))

    @staticmethod
    def activation_linear(x):
        return torch.ones_like(x)

    @staticmethod
    def activation_student_t(x):
        return x/(1+x**2)

    @staticmethod
    def activation_tanh(x):
        return torch.tanh(x)

    def __init__(self, config=None, file=None):
        super(TDV, self).__init__()

        self._is_TORCHSCRIPT_EXPORT:bool = TORCHSCRIPT_EXPORT # Hack for Torchscript export

        if (config is None and file is None) or \
            (not config is None and not file is None):
            raise RuntimeError('specify EITHER a config dictionary OR a `.pth`-file!')

        if not file is None:
            if not file.endswith('.pth'):
                raise ValueError('file needs to end with `.pth`!')
            checkpoint = torch.load(file)
            config = checkpoint['config']
            state_dict = checkpoint['model']
            self.tau = checkpoint['tau']
        else:
            state_dict = None
            self.tau = 1.0

        self.in_channels = config['in_channels']
        self.num_features = config['num_features']
        self.multiplier = config['multiplier']
        self.num_mb = config['num_mb']
        self.act = config['act']
        self.act_fin = config['act_fin']
        if 'zero_mean' in config.keys():
            self.zero_mean = config['zero_mean']
        else:
            self.zero_mean = True
        if 'num_scales' in config.keys():
            self.num_scales = config['num_scales']
        else:
            self.num_scales = 3

        self.psi = config['psi'] if 'psi' in config else 'linear'

        if self.psi == 'linear':
            self.out_pot = TDV.potential_linear
            self.out_act = TDV.activation_linear
        elif self.psi == 'student-t':
            self.out_pot = TDV.potential_student_t
            self.out_act = TDV.activation_student_t
        elif self.psi == 'tanh':
            self.out_pot = TDV.potential_tanh
            self.out_act = TDV.activation_tanh
        else:
            raise ValueError(f"Unknown value for psi:{self.psi}")

        # construct the regularizer
        self.K1 = Conv2d(self.in_channels, self.num_features, 3, zero_mean=self.zero_mean, invariant=False, bound_norm=True, bias=False)

        self.mb = torch.nn.ModuleList([MacroBlock(self.num_features, num_scales=self.num_scales, bound_norm=False, invariant=False, multiplier=self.multiplier, act=self.act,
                                                  act_fin=self.act_fin if i == self.num_mb-1 else self.act) 
                                        for i in range(self.num_mb)])

        self.KN = Conv2d(self.num_features, 1, 1, invariant=False, bound_norm=False, bias=False)

        if not state_dict is None:
            self.load_state_dict(state_dict)

    def _transformation(self, x, embedding: Optional[List[torch.Tensor]]=None):
        # extract features
        x = self.K1(x)
        # apply mb

        x_list = [x,] + (self.num_scales - 1)*[torch.zeros(1, device=x.device)]
        if embedding is not None:
            # print("use embedding here")
            for s in range(self.num_scales):
                x_list[s] = embedding[s] + x_list[s]

        for macro_block in self.mb:
            x_list = macro_block(x_list)

        out = self.KN(x_list[0])

        return out

    def _activation(self, x, lambda_mul: Optional[torch.Tensor]=None):
        # scale by the number of features
        # return torch.ones_like(x) / self.num_features
        act = self.out_act(x) / self.num_features
        if lambda_mul is not None:
            act = act * lambda_mul
        return act

    def _potential(self, x):
        # return x / self.num_features
        return self.out_pot(x) / self.num_features

    def _transformation_T(self, grad_out, embedding: Optional[List[torch.Tensor]]=None):
        # compute the output
        grad_x = self.KN.backward(grad_out)

        grad_x_list = [grad_x,] + (self.num_scales - 1)*[torch.zeros(1, device=grad_x.device)]
        # apply mb
        if self._is_TORCHSCRIPT_EXPORT:
            return self._transformation_T_helper(grad_x_list)
        else:
            return self._transformation_T_std(grad_x_list)

    def _transformation_T_helper(self, grad_x_list:List[torch.Tensor]):
        # VERY DIRTY HACK - see comment above in macroblock _forward_torchscript_helper  function.
        # TODO: Improve with future Pytorch versions
        if self.num_mb != 1:
            raise ValueError(f"TorchScript Export is currently only implemented for a fixed number of macroblocks num_mb=1 but found {self.num_mb} ")
        grad_x_list = self.mb[0].backward(grad_x_list)

        # extract features
        grad_x = self.K1.backward(grad_x_list[0])
        return grad_x

    @torch.jit.unused
    def _transformation_T_std(self, grad_x_list:List[torch.Tensor]):
        # apply mb
        for i in range(self.num_mb)[::-1]:
           grad_x_list = self.mb[i].backward(grad_x_list)

        # extract features
        grad_x = self.K1.backward(grad_x_list[0])
        return grad_x


    def energy(self, x, embedding=None, lambda_mul=None):
        x = self._transformation(x, embedding=embedding)
        if lambda_mul is not None:
            return self._potential(x) * lambda_mul
        else:
            return self._potential(x)

    def grad(self, x, get_energy: bool=False, embedding: Optional[List[torch.Tensor]]=None, lambda_mul: Optional[torch.Tensor]=None, apply_lambda_mul: bool=True):
        # compute the energy
        x = self._transformation(x, embedding=embedding)
        energy: Optional[torch.Tensor] = None
        if get_energy:
            # Get the Energy for visualization purposes (either with or without lambda_mul)
            energy = self._potential(x)
            if  apply_lambda_mul and lambda_mul is not None:
                energy = energy * lambda_mul
        # and its gradient
        x = self._activation(x, lambda_mul)
        grad = self._transformation_T(x)

        return energy, grad


class GradientTest(unittest.TestCase):
    
    def test_tdv_gradient(self):
        # setup the data
        x = np.random.rand(2,1,64,64)
        x = torch.from_numpy(x).cuda()

        # define the TDV regularizer
        config ={
            'in_channels': 1,
            'out_channels': 1,
            'num_features': 4,
            'num_scales': 3,
            'num_mb': 2,
            'multiplier': 2,
        }
        R = TDV(config).double().cuda()

        def compute_loss(scale):
            return torch.sum(R.energy(scale*x))
        
        scale = 1.
        
        # compute the gradient using the implementation
        grad_scale = torch.sum(x*R.grad(scale*x)).item()

        # check it numerically
        epsilon = 1e-4
        with torch.no_grad():
            l_p = compute_loss(scale+epsilon).item()
            l_n = compute_loss(scale-epsilon).item()
            grad_scale_num = (l_p - l_n) / (2 * epsilon)

        condition = np.abs(grad_scale - grad_scale_num) < 1e-3
        print(f'grad_scale: {grad_scale:.7f} num_grad_scale {grad_scale_num:.7f} success: {condition}')
        self.assertTrue(condition)


if __name__ == "__main__":
    unittest.test()

