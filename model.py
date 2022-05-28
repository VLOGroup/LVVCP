import torch
import torch.utils.checkpoint as cp
import logging
import numpy as np
import warnings

from ddr import TDV
from model_weight_net import get_mask_predictor
from common_utils.train_utils import add_parameter
from common_utils.utils import bcolors, get_min_value_lastdim, pt2np
from common_utils.utils_demo  import get_coclands_postfix
from common_utils.data_transformations import drop_prev_masks, propagate_old_masks
from common_utils.torch_script_logging import TorchScriptLogger, log_info, log_warning

from typing import Dict, Optional, List, Tuple


class Dataterm(torch.nn.Module):
    """
    Basic dataterm function
    """
    def __init__(self, config):
        super(Dataterm, self).__init__()

    @torch.jit.unused
    def forward(self, x, *args):
        raise NotImplementedError

    def energy(self):
        raise NotImplementedError

    def prox(self, x, *args):
        raise NotImplementedError

    def grad(self, x, *args):
        raise NotImplementedError
        
class L2DenoiseDataterm(Dataterm):
    """
    D(x) = lambda * | x - z |²
    """
    def __init__(self, config):
        super(L2DenoiseDataterm, self).__init__(config)

    def energy(self, x, z, lmbda:Optional[torch.Tensor]=None):
        energy = 0.5*(x-z)**2
        energy = energy if lmbda is None else energy * lmbda
        return energy

    def prox(self, x, z, taulambda):
        return (x + taulambda * z) / (1 + taulambda) 

    def grad(self, x, z):
        return x-z

    def grad_lambda(self, x, z):
        """ d(energy)/d(tau lambda)
            If lmbda uses broadcasting, the gradients need to be summed here
        """
        return self.energy(x, z, lmbda=None)

class ColcandGenerator(torch.nn.Module):
    """
    Base class for yielding color candidates
    """
    def __init__(self ):
        super(ColcandGenerator, self).__init__()
        pass

    def get_cands_and_energy(self, data:Dict[str, torch.Tensor], xk_cur:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ get all candidates and the according energy"""
        raise not ImportError()

    def get_best_cand(self, data:Dict[str, torch.Tensor], xk_cur:torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:
        """ get the current best candidate based on the current iteration"""
        raise not ImportError()

class ColCandFlowWarped(ColcandGenerator):
    """
    Returns color candidates and according energy, based on optical flow
    """
    def __init__(self, data_key:str, colcand_selectby_ab_nLab:bool):
        super(ColCandFlowWarped, self).__init__()
        self.data_key = data_key
        self.dataterm = L2DenoiseDataterm(None)
        self.colcand_selectby_ab_nLab = colcand_selectby_ab_nLab

    def get_cands_and_energy(self, data:Dict[str, torch.Tensor], xk_cur:torch.Tensor):
        """ get all candidates and the according energy"""
        cand_best = data['i0cw'].clone()  
        cand_best[:,0:1] = data['i1L'][:,0:1].clone() # replace luminance with initial luminance
        cands = cand_best[...,None] # [N,C,H,W,1]
        
        if self.colcand_selectby_ab_nLab:  # Select by Lab or ab distance energy
            selection_energy = self.dataterm.energy( xk_cur[:,1:3,...,None], cands[:,1:3,...]).sum(dim=1,keepdim=True) #NCHWK
        else:
            selection_energy = self.dataterm.energy( xk_cur[...,None], cands).sum(dim=1,keepdim=True) #NCHWK
        return cands, selection_energy

    def get_best_cand(self, data:Dict[str, torch.Tensor], xk_cur:torch.Tensor):
        """ get the current best candidate based on the current iteration
            For flow this is iteration independent. """
        cand_best = data['i0cw'].clone()  
        cand_best[:,0:1] = data['i1L'][:,0:1].clone() # replace luminance with initial luminance
        
        ret_val = {}
        ret_val[f"i1c_colcand{self.data_key}_prox"  ] = cand_best.detach().cpu() # save for visualization
        ret_val[f"i1c_colcand{self.data_key}_change"] = (data[self.data_key].detach() - cand_best.detach()).abs().mean(dim=1,keepdim=True).cpu()
        return cand_best, ret_val

    @torch.jit.unused
    def __repr__(self):
        return f"ColCandFlowWarped(data_key={self.data_key})"

class ColCandInit(ColcandGenerator):
    """
    Returns color candidates and according energy, using the initial fused image
    """
    def __init__(self, data_key:str, colcand_selectby_ab_nLab:bool):
        super(ColCandInit, self).__init__()
        self.data_key = data_key
        self.dataterm = L2DenoiseDataterm(None)
        self.colcand_selectby_ab_nLab = colcand_selectby_ab_nLab

    def get_cands_and_energy(self, data:Dict[str, torch.Tensor], xk_cur:torch.Tensor):
        """ get all candidates and the according energy"""
        cand_best = data['i1c_init'].clone()  
        cand_best[:,0:1] = data['i1L'][:,0:1].clone()
        cands = cand_best[...,None] # [N,C,H,W,1]
        
        if self.colcand_selectby_ab_nLab:  # Select by Lab or ab distance energy
            selection_energy = self.dataterm.energy( xk_cur[:,1:3,...,None], cands[:,1:3,...]).sum(dim=1,keepdim=True) #NCHWK
        else:
            selection_energy = self.dataterm.energy( xk_cur[...,None], cands).sum(dim=1,keepdim=True) #NCHWK
        return cands, selection_energy

    def get_best_cand(self, data:Dict[str, torch.Tensor], xk_cur:torch.Tensor):
        """ get the current best candidate based on the current iteration
            For flow this is iteration independent. """
        cand_best = data['i1c_init'].clone()  
        cand_best[:,0:1] = data['i1L'][:,0:1].clone()
        
        ret_val = {}
        ret_val[f"i1c_best_{self.data_key}_prox"  ] = cand_best.detach().cpu() # save for visualization
        ret_val[f"i1c_best_{self.data_key}_change"] = (data[self.data_key].detach() - cand_best.detach()).abs().mean(dim=1,keepdim=True).cpu()
        return cand_best, ret_val

    @torch.jit.unused
    def __repr__(self):
        return f"ColCandInit(data_key={self.data_key}, colcand_selectby_ab_nLab={self.colcand_selectby_ab_nLab})"

class ColCandMultiL2(ColcandGenerator):
    """
    Returns color candidates and according energy for multiple color candidates
    """
    def __init__(self, data_type:str, colcand_selectby_ab_nLab:bool):
        super(ColCandMultiL2, self).__init__()
        self.data_type = data_type
        self.data_key = f"colcands1{data_type}"
        self.data_key_best = f"colcands1_best{data_type}"
        self.colcand_selectby_ab_nLab = colcand_selectby_ab_nLab
        self.dataterm = L2DenoiseDataterm(None)

    def get_cands_and_energy(self, data:Dict[str, torch.Tensor], xk_cur:torch.Tensor):
        """ get all candidates and the according energy"""
        cands = data[self.data_key]
        if len(cands.shape) != 5 : raise ValueError(f"expected shape [N,K,C,H,W] but found {cands.shape} for {self.data_key} ")
        cands =  [cands[:,i] for i in range(cands.shape[1])] # [N,K,C,H,W]
        cands = torch.stack(cands,dim=-1)   

        if self.colcand_selectby_ab_nLab:  # Select by Lab or ab distance energy
            selection_energy = self.dataterm.energy( xk_cur[:,1:3,...,None], cands[:,1:3,...]).sum(dim=1,keepdim=True) #NCHWK
        else:
            selection_energy = self.dataterm.energy( xk_cur[...,None], cands).sum(dim=1,keepdim=True) #NCHWK
        return cands, selection_energy

    def get_best_cand(self, data:Dict[str, torch.Tensor], xk_cur:torch.Tensor):
        """ get the current best candidate based on the current position"""
        cands, selection_energy = self.get_cands_and_energy(data, xk_cur)
        cand_best, energy_cand_best = get_min_value_lastdim(cands, selection_energy, get_best_cond_val=True)
        ret_val = {}
        ret_val[f"i1c_best{self.data_type}_prox"  ] = cand_best.detach().cpu() # save for visualization
        ret_val[f"i1c_best{self.data_type}_change"] = (data[self.data_key_best].detach() - cand_best.detach()).abs().mean(dim=1,keepdim=True).cpu()
        return cand_best, ret_val

    @torch.jit.unused
    def __repr__(self):
        return f"ColCandMultiL2(data_type={self.data_type}, colcand_selectby_ab_nLab={self.colcand_selectby_ab_nLab})"


class L2DenoiseVarcntcandDataterm(Dataterm):
    def __init__(self, config):
        super(L2DenoiseVarcntcandDataterm, self).__init__(config)

    def energy(self, x, z_lambda_mask_lst: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] ):
        energy = torch.zeros_like(x)
        for i, ( z, lmbda, mask) in  enumerate (z_lambda_mask_lst):
            assert x.shape == z.shape, f"x, z must have the same shapes for all examples!"
            energy = energy + 0.5 * lmbda * mask * (x-z) ** 2
        return energy

    def prox(self, x, z_taulambda_mask_lst: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
        num   = x
        denom = torch.ones_like(x)
        for i, ( z, taulambda, mask) in  enumerate (z_taulambda_mask_lst):
            assert x.shape==z.shape, f"x, z must have the same shapes for all examples!"
            taulambda_mask = taulambda * mask
            num   = num   + taulambda_mask * z   #  (x +   sum( (tau*lambda_i) * mask_i))  * z_i )
            # -------------------------------    #  ----------------------------------------------
            denom = denom + taulambda_mask       #  (1 +   sum( (tau*lambda_i) * mask_i))        )
        x_prox = num / denom 

        return x_prox

    def grad_x(self, x, z_lambda_mask_lst):
        grad_x = 0
        for i, (z, lmbda, mask) in  enumerate (z_lambda_mask_lst):
            assert x.shape == z.shape, f"x, z must have the same shapes for all examples!"
            grad_x = grad_x + lmbda * mask * (x-z)
        return grad_x

    def grad_masks(self, x, z_lambda_mask_lst):
        """ d(energy)/d(mask0), d(energy)/d(maskcand)
            If a mask uses broadcasting, the gradients need to be summed here
        """
        grads = []
        for i, (z, lmbda, mask) in  enumerate (z_lambda_mask_lst):
            assert x.shape == z.shape, f"x, z must have the same shapes for all examples!"
            grads.append( 0.5 * lmbda * (x-z)**2 )
        return grads
    
    def grad_masks_nolambda(self, x, z_mask_lst):
        """ d(energy)/d(mask0), d(energy)/d(maskcand)
            If mask uses broadcasting, the gradients need to be summed here

            returns both parts seperately, such that they can later be multiplied by taulambda1 and taulambda4 seperately
        """
        grads = []
        for i, (z, mask) in  enumerate (z_mask_lst):
            assert x.shape == z.shape, f"x, z must have the same shapes for all examples!"
            grads.append( 0.5 *  (x-z)**2 )
        return grads



class ConsistentRecolouringLab(Dataterm):
    """
    This dataterm uses the multiple candidates weighted by the according lambda terms.
    It uses the approximation that candidates are independent from each other.

    D(x) =  lambda_M * mask_M | x - z_M | ^2 + 
            min_{z_{G,K}} lambda_G * mask_G | x - z_GK | ^2  +
            min_{z_{L,K}} lambda_L * mask_L | x - z_LK | ^2  +
    """
    D_use_cand_flow_warped : bool = False
    D_use_cand_init_image  : bool = False
    D_use_cand_global      : bool = False
    D_use_cand_local       : bool = False

    def __init__(self, config):
        super(ConsistentRecolouringLab, self).__init__(config)

        self.logger = logging.getLogger("main-logger")
        self.config = config

        self.gen_colcand_flow: Optional[ColCandFlowWarped] = None
        self.gen_colcand_init: Optional[ColCandFlowWarped] = None
        self.gen_colcand_global: Optional[ColCandMultiL2] = None
        self.gen_colcand_local: Optional[ColCandMultiL2] = None

        # MultiCand selection is done via 'ab' distance or Lab distance
        self.colcand_selectby_ab_nLab = False # backward compatibility
        if 'colcand_selectby' in config:
            if  config['colcand_selectby'] == "ab":
                self.colcand_selectby_ab_nLab = True
            elif config['colcand_selectby'] == "Lab":
                self.colcand_selectby_ab_nLab = False
                raise ValueError(f" colcand_selectby should be 'ab'")
            else:
                raise ValueError(f"{ config['colcand_selectby'] } must be in ['ab','Lab']")
            self.logger.info(f"Dataterm: selecting best candidate based on colcand_selectby 'ab':{self.colcand_selectby_ab_nLab} not 'Lab' distance energy")

        
        self.use_sep_cand_lambdas = True if ('use_sep_cand_lambdas' in self.config and self.config['use_sep_cand_lambdas']) else False

        if "D_ab_norm" in config:
            if (config["D_ab_norm"] == "l2proxVarCandCnt"  ):
                self.datatermMulti = L2DenoiseVarcntcandDataterm(self.config) # allows for a variable amount of input candidates
                
                refs_allowed = ['flow','glob','loc','i1c_init']
                dataterm_ref_cands_set = set(config['dataterm_ref_cands'])
                if not dataterm_ref_cands_set.issubset(refs_allowed):
                    raise ValueError(f"Unknown parameter for dataterm_ref_cands:{config['dataterm_ref_cands']} mut be in {refs_allowed} ")

                if 'i1c_init' in dataterm_ref_cands_set:
                    if 'flow' in dataterm_ref_cands_set:
                        raise ValueError(f"Cannot use  'i1c_init' and 'flow' at the same time for 'dataterm_ref_cands'")
                    self.D_use_cand_init_image = True
                    self.gen_colcand_init = ColCandInit('i1c_init', self.colcand_selectby_ab_nLab)
                    self.logger.info(f"L2DenoiseVarcntcandDataterm: Adding i1c_init (initially fused result)")
                if 'flow' in dataterm_ref_cands_set:
                    self.D_use_cand_flow_warped = True
                    self.gen_colcand_flow = ColCandFlowWarped('i0cw', self.colcand_selectby_ab_nLab)
                    self.logger.info(f"L2DenoiseVarcntcandDataterm: Adding i0cw (i0 warped with flow)")
                if 'glob' in dataterm_ref_cands_set:
                    if self.use_sep_cand_lambdas:
                        add_parameter(self, 'tau_lambda_cand', self.config)
                        self.logger.info(f"Dataterm: Using Separate Lambdas for the different candidates from '{self.use_sep_cand_lambdas}'")
                    self.D_use_cand_global = True
                    self.candglob_postfix = get_coclands_postfix('glob')
                    self.gen_colcand_global = ColCandMultiL2(self.candglob_postfix, self.colcand_selectby_ab_nLab)
                    self.logger.info(f"L2DenoiseVarcntcandDataterm: Adding matchign based global candidate")
                if 'loc' in dataterm_ref_cands_set:
                    if self.use_sep_cand_lambdas:
                        add_parameter(self, 'tau_lambda_cand3', self.config)
                        self.logger.info(f"Dataterm: Adding  'tau_lambda_cand3' as new parameter with config {self.config['tau_lambda_cand3']}")

                    self.D_use_cand_local = True
                    self.candloc_postfix = get_coclands_postfix('loc')
                    self.gen_colcand_local  = ColCandMultiL2(self.candloc_postfix, self.colcand_selectby_ab_nLab)
                    self.logger.info(f"L2DenoiseVarcntcandDataterm: Adding matchign based local candidate")
            else:
                raise ValueError(f"'D_ab_norm'='{config['D_ab_norm']}' not implemented!")

        if 'denoise_mask' in config and config['denoise_mask']: 
            raise NotImplementedError(f"'denoise_mask' option not implemented here")

        # Check if Regularizer uses additional inputs => changes size of X
        for old_cfg_key in ["TDV_use_mask", "TDV_use_warped_gray","TDV_use_fdelta","TDV_use_mask_cand","TDV_use_mask_cand3",
                            "TDV_use_mask_none", "TDV_use_i1g_ab_init"]:
            if old_cfg_key in config and config[old_cfg_key]:
                raise ValueError(f"{old_cfg_key} not implemented")


    def energy(self, xk_c_m1, tau , grad_R,  z, tau_lambda):
        raise NotImplementedError()

    def prox(self, xk_c_m1, tau , grad_R,  z: Dict[str, torch.Tensor], tau_lambda, get_energy: bool=False):
        """Proximal mapping combining multiple color candidates """
        #Perform proximal step on Dataterm
        energies_dat: Dict[str, torch.Tensor] = {}

        # compute gradient step on Regularizer internally - gives access to previous value of xk_c_m1 and update xk_c
        xk_c = xk_c_m1 - tau * grad_R

        ###################################
        # BEGIN Proxmap for ab space
        # Build the input for the Dataterm with variable amount of input data
        vis = {}
        z_taulambda_mask_lst: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        cand_est_xk_c_prox_in = xk_c
        if self.gen_colcand_flow is not None:
            candidate, vis_tmp = self.gen_colcand_flow.get_best_cand(z, cand_est_xk_c_prox_in)
            z_taulambda_mask_lst.append((candidate, tau_lambda, z['matched10_dat_flow']))
            vis.update(vis_tmp)
        elif self.gen_colcand_init is not None:
            candidate, vis_tmp = self.gen_colcand_init.get_best_cand(z, cand_est_xk_c_prox_in)
            z_taulambda_mask_lst.append((candidate, tau_lambda,  z['matched10_dat_flow']))
            vis.update(vis_tmp)

        if self.gen_colcand_global is not None:
            candidate, vis_tmp = self.gen_colcand_global.get_best_cand(z, cand_est_xk_c_prox_in)
            tau_lambda_cand = (self.tau_lambda_cand / self.S)  if self.use_sep_cand_lambdas else tau_lambda
            z_taulambda_mask_lst.append((candidate, tau_lambda_cand, z['matched10_dat_glob']))
            vis.update(vis_tmp)

        if self.gen_colcand_local is not None:
            candidate, vis_tmp = self.gen_colcand_local.get_best_cand(z, cand_est_xk_c_prox_in)
            tau_lambda_cand3 = (self.tau_lambda_cand3 / self.S) if self.use_sep_cand_lambdas else tau_lambda
            z_taulambda_mask_lst.append((candidate, tau_lambda_cand3, z['matched10_dat_loc']))
            vis.update(vis_tmp)

        xk_c_prox = self.datatermMulti.prox(x=xk_c, z_taulambda_mask_lst=z_taulambda_mask_lst)

        
        # 2nd Run - For Visualization of Energy only:
        if get_energy:
            with torch.no_grad():
                z_lambda_mask_lst: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = [] # List of [z, lambda, mask]
                for (candidate, taulambda, mask) in z_taulambda_mask_lst:
                    z_lambda_mask_lst.append((candidate, taulambda / tau, mask))
                # Compute the energy for the paths taken (i.e. the selected candidates)
                energies_dat['energy_datall_xkprox'] = self.datatermMulti.energy( x=xk_c  , z_lambda_mask_lst=z_lambda_mask_lst)
                energies_dat['energy_datall_xk0'   ] = self.datatermMulti.energy( x=xk_c_m1,z_lambda_mask_lst=z_lambda_mask_lst)            

        # Keep Luminescense => Hard Projection onto original GrayScale image
        xk_c_prox[:,0:1] = z['i1c_init'][:,0:1]  # NCHW keep luminescence

        # END Proxmaps ab space
        ###################################

        for k,v in vis.items():
            z[k] = v

        return xk_c_prox.contiguous(), energies_dat

    def grad(self, x, z: Dict[str, torch.Tensor],) -> torch.Tensor:
        raise NotImplementedError()
        return x  # dummy for TorchScript



class VNet(torch.nn.Module):
    """
    Variational Network
    """

    def __init__(self, config, efficient=False):
        super(VNet, self).__init__()

        self.efficient = efficient
        
        self.S = config['S']

        # setup the stopping time
        add_parameter(self, 'T', config)
        # setup the regularization parameter * stopping time
        add_parameter(self, 'taulambda', config)

        # setup the regularization
        R_types = {
            'tdv': TDV,
        }
        self.R = R_types[config['R']['type']](config['R']['config'])

        # setup the dataterm
        self.use_prox = config['D']['config']['use_prox']
        D_types = {
            'video_denoise_lab':ConsistentRecolouringLab,         
        }
        self.D = D_types[config['D']['type']](config['D']['config'])
        self.D.S = 0                  # placeholders need to be updated before calling D.prox
        self.D.tau = torch.tensor(0)  # placeholders need to be updated before calling D.prox

        self.seperate_stages_count:Optional[int] = None

    def set_separate_stages_count(self, stages_count):
        """ Use this function to change the amount of refinement steps, but keeping T/S as trained"""
        self.seperate_stages_count = stages_count
        log_info(f"Using a stages count that is independent from S! S={self.S}, stages={self.seperate_stages_count}")


    def forward(self, x, z:Dict[str, torch.Tensor], get_grad_R: bool=False, get_gradR_L: bool=False, embedding: Optional[List[torch.Tensor]]=None, get_energy: bool=False, get_tau_Lgrad: bool=False):

        if self.seperate_stages_count is None:
            stages = self.S+1
        else:
            stages = self.seperate_stages_count+1 # Seperate stages from T for inference

        # x_all = x.new_empty((stages,*x.shape))
        x_all = [x.new_empty(x.shape),]*(stages)
        x_all[0] = x  # element 0 is initial image

        grad_R_all: Optional[torch.Tensor]   = None 
        if get_grad_R:
            grad_R_all = x.new_empty((stages-1,) +x.shape)

        # define the step size
        tau = self.T / self.S
        taulambda = self.taulambda / self.S
        L = []
        tau_Lgrad = []
        energies_reg : List[ torch.Tensor] = []
        energies_dat : List[Dict[str, torch.Tensor]] = []
        Lipschitz: Dict[str, List[torch.Tensor]] = {}
        
        lambda_tdv_mul = z['lambda_tdv_mul'] if 'lambda_tdv_mul' in z else None
        for s in range(1, stages):  # compute a single step
            # 1. Compute Gradient of Regularizer (wrap into gradient checkpointing if desired)
            energy_reg, grad_R = self.R.grad( x,
                            embedding=embedding,
                            lambda_mul=lambda_tdv_mul,
                            apply_lambda_mul=False, # Do not apply lambda_mul to the energy here, so that we can visualize it, ot it later
                            get_energy=get_energy)
                            
            if energy_reg is not None:
                energies_reg += [energy_reg]
            

            ################
            # 2. Compute update on Dataterm 
            if self.use_prox:
                self.D.S = self.S
                self.D.tau = tau
                x, en_dat = self.D.prox(x, tau , grad_R, z, taulambda, get_energy=True)
                energies_dat.append(en_dat)
            else:
                x = x - tau * grad_R - taulambda * self.D.grad(x, z)

            if get_grad_R and (grad_R_all is not None):
                grad_R_all[s-1] = grad_R
            x_all[s] = x

        
        if stages == 1:
            Lipschitz.clear()
        if stages == 2:
            L = [torch.zeros([x.shape[0],1,1,1])]
            Lipschitz.clear()
        if stages == 2:
            tau_Lgrad = [torch.zeros([x.shape[0],1,1,1])]

        return x_all, Lipschitz, energies_reg, energies_dat

    def set_end(self, s):
        assert 0 < s
        self.S = s

    def extra_repr(self):
        s = "S={S}"
        return s.format(**self.__dict__)


class Gen_Init_Image(torch.nn.Module):
    """This class sets up the intial re-colouring image"""
    def __init__(self, config):
        super(Gen_Init_Image,self).__init__()
        self.config = config

        self.use_warped_flow:bool  = False
        self.use_matched_glob:bool = False
        self.use_matched_loc:bool  = False

        if type(config['init_image']) == list: #  e.g. ['flow', 'glob', 'loc']
            init_image_allowed = ['flow','glob','loc']
            s_init_image = set(config['init_image'])
            if not s_init_image.issubset(init_image_allowed):
                raise ValueError(f"unkown parameter for init_image{config['init_image']} must be in {init_image_allowed} ")
            if 'flow' in s_init_image:
                self.use_warped_flow = True
            if 'glob' in s_init_image:
                self.use_matched_glob = True
            if 'loc' in s_init_image:
                self.use_matched_loc = True
        else:
            raise ValueError(f"Unkown or old fromat for init_image: {config['init_image']}")

        log_info(f"GenInitImage New:"
                 f"flow={self.use_warped_flow}, "
                 f"glob={self.use_matched_glob}, "
                 f"loc={self.use_matched_loc}" )
                 
    def forward(self, data:Dict[str, torch.Tensor]):
        i1c_init = torch.full_like(data['i0c'],  0.5)  # Initialize with an image full 0f 0.5, which is zero in the normalized space
        if self.use_warped_flow:
            if not 'i0cw' in data: raise ValueError(f"Flow is not present input data, check settings for dataloader")
            i1c_init = i1c_init + data['matched10_img_flow'] * (data['i0cw'               ] -0.5)
            data['i1c_colcand_flow_init'] = data['i0cw']
            log_info(f"GenInitImage: Using Flow warped image")
        if self.use_matched_glob:
            if not 'colcands1_best_glob' in data: raise ValueError(f"Global Canddiates are not present input data, check settings for dataloader")
            i1c_init = i1c_init + data['matched10_img_glob'] * (data['colcands1_best_glob'] -0.5)
            data['i1c_colcand_glob_init'] = data['colcands1_best_glob']
            log_info(f"GenInitImage: Using Global Matched image")
        if self.use_matched_loc:
            if not 'colcands1_best_loc' in data: raise ValueError(f"Local Canddiates are not present input data, check settings for dataloader")
            i1c_init = i1c_init + data['matched10_img_loc' ] * (data['colcands1_best_loc' ] -0.5)
            data['i1c_colcand_loc_init'] = data['colcands1_best_loc']
            log_info(f"GenInitImage: Using Local Matched image")

        i1c_init[:,0:1] = data['i1L'][:,0:1].clone()
        data['i1c_init'] = i1c_init



class VideoRecolourer(torch.nn.Module):
    def __init__(self,config, logger):
        super(VideoRecolourer, self).__init__()
        self.config = config
        self.vn = VNet(config, efficient=config['R']['config']['efficient'])
        self.gen_init_image = Gen_Init_Image(config['D']['config'])
        self.logger = logger

        if 'TDV_use_mask' in config['D']['config'] and  config['D']['config']['TDV_use_mask']:
            raise NotImplemented(f" TDV_use_mask not implemented")

        self.mask_pred = get_mask_predictor(config['MaskPredictor'])

        self.get_energy=False
        if ("get_energy" in config['D'] and config['D']['get_energy'] or
            "get_energy" in config['D']['config'] and config['D']['config']['get_energy'] ):
            self.get_energy=True

        self.propagate_energy_multiplied = False
        if 'propagate_energy_multiplied' in config['D']['config'] and config['D']['config']['propagate_energy_multiplied']:
            self.propagate_energy_multiplied = True
            log_info(f"Daterm: energy_reg = energy_reg * lambda_mul, This is visualized and propaged to MaskCNN 'propagate_energy_multiplied':{self.propagate_energy_multiplied}")

        self.drop_prev_warp_dataterm_masks_S0 = False
        if 'drop_prev_warp_dataterm_masks_S0' in config['D']['config'] and config['D']['config']['drop_prev_warp_dataterm_masks_S0']:
            self.drop_prev_warp_dataterm_masks_S0 = True
            log_info(f"Daterm: Stop usage of previous warped Dataterm masks as long as S=0 'drop_prev_warp_dataterm_masks_S0':{self.drop_prev_warp_dataterm_masks_S0}")

        self.drop_prev_warp_dataterm_masks_always = False
        if 'drop_prev_warp_dataterm_masks_always' in config['D']['config'] and config['D']['config']['drop_prev_warp_dataterm_masks_always']:
            self.drop_prev_warp_dataterm_masks_always = True
            log_info(f"Daterm: Stop usage of previous masks (warped, dataterm, init image, reg, tdv), 'drop_prev_warp_dataterm_masks_always':{self.drop_prev_warp_dataterm_masks_always}")


        self.bypass_mask_prediction = False

    def set_separate_stages_count(self, stages):
        """ Use this function to change the amount of refinement steps, but keeping step size tau=T/S as trained"""
        self.vn.set_separate_stages_count(stages)

    def get_S(self):
        """ Get the amount of variational refinement steps, will also affect step size tau=T/S"""
        return self.vn.S
    
    def set_S(self, S):
        """ Set the amount of variational refinement steps, will also affect step size tau=T/S"""
        self.logger.info(f"Model Update! Changes S:{self.vn.S}->{S}")
        self.vn.S = S
        self.vn.D.S = S

    @torch.jit.unused
    def build_vis_res(self, data: Dict[str, torch.Tensor], x_lst: List[torch.Tensor], embedding: Optional[List[torch.Tensor]], Lipschitz: Dict[str, List[torch.Tensor]], energy_reg: List[torch.Tensor], energy_dat: List[Dict[str, torch.Tensor]] ) -> Dict[str, torch.Tensor]:
        """ 
        Returns additional results for visualization purposes.
        Implementing the nested datastructures did not work with pt1.7, hence a dummy definition was used for the required type hinting for TorchScript - hence this function cannot be torch scripted.
        """
        res = {'i1c': x_lst}
        res["Lipschitz"] = Lipschitz
        res["S"] = self.vn.S

        res['i1c_est'] = res['i1c']
        if 'matched10_dat_flow' in res:
            res['matched_est'] = res['matched10_dat_flow']
        else:  # if no refinement of mask is desired keep original mask
            res['matched_est'] = [data['matched10_img_flow'],] * len(res['i1c_est'])
            res['matched10'] =  res['matched_est']

        if hasattr(self.mask_pred,"_use_embedding_in_maskcnn") and self.mask_pred._use_embedding_in_maskcnn:
            res['maskcnn_embedding'] = embedding

        if len(energy_reg) > 0:
            for key in energy_dat[0]:
                res[key] = [energy_dat[i][key] for i in range(len(energy_dat))]
            #res['energy_dat_est'] = res['energy_dat'] = energy_dat[-1]
            energy_reg_fin = energy_reg[-1]
            if 'lambda_tdv_mul' in res and self.propagate_energy_multiplied:
                energy_reg_fin *= res['lambda_tdv_mul'][0]

            energy_reg_max = torch.stack([en.detach().abs().flatten(1,3).max(dim=-1)[0][:,None,None,None] for en in energy_reg],dim=0).max(dim=0)[0]

            res['energy_reg_fin' ] = [energy_reg_fin]
            res['energy_reg_max' ] = [energy_reg_max]
            res['energy_reg'     ] =  energy_reg     # [en.detach()/res['energy_max'] for en in energy]
            res['en_scalar_reg'          ] = [en.mean(dim=(1,2,3),keepdim=True) for en in res['energy_reg'] ]
            res['en_scalar_datall_xk0'   ] = [en.mean(dim=(1,2,3),keepdim=True) for en in res['energy_datall_xk0'] ]
            res['en_scalar_datall_xkprox'] = [en.mean(dim=(1,2,3),keepdim=True) for en in res['energy_datall_xkprox'] ]
            res['en_scalar_cmbd_xk0'     ] = [er + ed for er,ed in zip(res['en_scalar_reg'],res['en_scalar_datall_xk0'])]
            res['en_scalar_cmbd_xkprox'  ] = [er + ed for er,ed in zip(res['en_scalar_reg'],res['en_scalar_datall_xkprox'])]

        return res


    def forward(self, data:Dict[str, torch.Tensor], data_prev:Optional[Dict[str, torch.Tensor]]=None, get_res:bool=False):
        # Propagate old masks if available
        if data_prev is not None:
            propagate_old_masks(data, data_prev)

        # Use drop Previous Masks if it is desired to have independent Frames
        if self.drop_prev_warp_dataterm_masks_always:
            data = drop_prev_masks(data)

        # compute the initial confidence mask, either heuristic or CNN
        embedding =None
        if self.bypass_mask_prediction:
            log_warning(f"Bypassing Mask Prediction")
        else:
            self.mask_pred(data) 
        # Generate the initial image by warping the old + filling up occlusions by likely candidates

        self.gen_init_image(data) # Let the MaskCNN build an Image and then destroy it in some regions

        x_refined_lst, Lipschitz, energy_reg, energy_dat = self.vn(data['i1c_init'] , data, get_gradR_L=get_res, get_energy=self.get_energy)
        x_refined = x_refined_lst[-1]

        vis_res = {}
        if get_res:
            vis_res = self.build_vis_res(data, x_refined_lst, embedding, Lipschitz, energy_reg, energy_dat)

        return x_refined, vis_res

