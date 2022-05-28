import torch
import torch.nn.functional as F
import logging
import numpy as np
import warnings

from common_utils.train_utils import add_parameter, get_parameter_v2
from ddr import  MaskPredictor
from ddr.conv import Conv2d
from common_utils.utils import bcolors, get_min_value_lastdim
from common_utils.flow_utils import flow_warp
from common_utils.utils_demo import get_coclands_postfix

from pprint import pformat

from typing import Dict, List, Union, Tuple, Optional

from common_utils.torch_script_logging import TORCHSCRIPT_EXPORT, log_info


####################################################################################################
# Mask Predictor Classes
def  get_mask_predictor(config):
    if config['type'] == 'MaskPredictorCnnVar': 
        return MaskPredictorCnnVar(config['config'])
    if config['type'] == 'MaskPredictorCnnVarCmbdHead': 
        return MaskPredictorCnnVarCmbdHead(config['config'])
    elif config['type'] == 'heuristic_hard':
        return MaskPredictorHardHeuristic(config['config'])
    elif config['type'] == 'heuristic_soft':
        return MaskPredictorSoftHeuristic(config['config'])
    elif config['type'] == 'keep_all':
        return MaskPredictorKeepAll(config['config'])
    elif config['type'] == 'ONLY_Local':
        return MaskPredictorKeepAllLocalCands(config['config'])
    elif config['type'] == 'ONLY_Global':
        return MaskPredictorKeepAllGlobalCands(config['config'])
    elif config['type'] == 'Global_TDV':
        return MaskPredictorKeepAllGlobalCandsAndTDV({})
    elif config['type'] == 'ONLY_Flow':
        return MaskPredictorKeepAllFlow(config['config'])
    elif config['type'] == 'ORACLE_Local':
        return MaskPredictorOracleLocalCands(config['config'])
    elif config['type'] == 'ORACLE_Global':
        return MaskPredictorOracleGlobalCands(config['config'])
    elif config['type'] == 'ORACLE_GlobalFlow':
        return MaskPredictorOracleGlobalFlow(config['config'])
    elif config['type'] == 'ORACLE_LocalFlow':
        return MaskPredictorOracleLocalFlow(config['config'])
    elif config['type'] == 'ORACLE_GlobalLocalFlow':
        return MaskPredictorOracleGlobalLocalFlow(config['config'])
    else:
        raise ValueError(f"Wrong config type for 'MaskPredictor' {config}")

class MaskPredictorIF(torch.nn.Module):
    """ Interface definition for MaskPredictor class,  
        Predicts the Initial Confidence given the data
    """
    def __init__(self):
        super(MaskPredictorIF, self).__init__()
    def _predict(self, data: Dict[str, torch.Tensor]) -> Optional[List[torch.Tensor]]:
        raise NotImplementedError('Implement in derived class! This is just the interface')
    def forward(self, data: Dict[str, torch.Tensor]) -> Optional[List[torch.Tensor]]:
        return self._predict(data)
        # This part fills up 'matched_01' with the initial confidence mask

class MaskPredictorHardHeuristic(MaskPredictorIF):
    def __init__(self, config):
        super(MaskPredictorHardHeuristic, self).__init__()
    def _predict(self, data):
        data['matched10'] = data['matched10_hard']   # Use hard mask UnFlow

class MaskPredictorKeepAll(MaskPredictorIF):
    def __init__(self, config):
        super(MaskPredictorKeepAll, self).__init__()
    def _predict(self, data):
        data['matched10'] = torch.ones_like(data['matched10_hard'])  #keep all pixels

##############################################################################################
# BEGIN ORACLE SECTION
class MaskPredictorOracleLocalCands(MaskPredictorIF):
    def __init__(self, config):
        super(MaskPredictorOracleLocalCands, self).__init__()
        log_info(f"Using Oracle for Mask Prediction CNN (Local)")
    def _predict(self, data):
        one  = torch.ones_like(data['matched10_hard'])  
        zero = torch.zeros_like(data['matched10_hard'])        
        data['matched10_img_flow'] = zero
        data['matched10_img_glob'] = zero
        data['matched10_img_loc' ] = one
        data['matched10_img_none'] = zero
        data['matched10'] = data['matched10_img_flow']  # backward compatibility

        # Select Best Candidate by Oracle Distance from GT:        
        cands =  [data['colcands1_loc'][:,i] for i in range(data['colcands1_loc'].shape[1])] # [N,K,C,H,W]
        cands = torch.stack(cands,dim=-1)
        distance = cands-data['i1c'][...,None] # [N,K,C,H,W]        
        distance_ab_l2 = torch.norm(distance[:,1:3], dim=1, p=2, keepdim=True) # [N,K,1,H,W]
        cand_best, _ = get_min_value_lastdim(cands, distance_ab_l2, get_best_cond_val=True)
        data['colcands1_best_loc'] = cand_best

        return None
        
class MaskPredictorOracleGlobalCands(MaskPredictorIF):
    def __init__(self, config):
        super(MaskPredictorOracleGlobalCands, self).__init__()
        log_info(f"Using Oracle for Mask Prediction CNN (Global)")
    def _predict(self, data):
        one  = torch.ones_like(data['matched10_hard'])  
        zero = torch.zeros_like(data['matched10_hard'])        
        data['matched10_img_flow'] = zero
        data['matched10_img_glob'] = one
        data['matched10_img_loc' ] = zero
        data['matched10_img_none'] = zero
        data['matched10'] = data['matched10_img_flow']  # backward compatibility

        # Select Best Candidate by Oracle Distance from GT:        
        cands =  [data['colcands1_glob'][:,i] for i in range(data['colcands1_glob'].shape[1])] # [N,K,C,H,W]
        cands = torch.stack(cands,dim=-1)
        distance = cands-data['i1c'][...,None] # [N,K,C,H,W]
        distance_ab_l2 = torch.norm(distance[:,1:3], dim=1, p=2, keepdim=True) # [N,K,1,H,W]
        cand_best, _ = get_min_value_lastdim(cands, distance_ab_l2, get_best_cond_val=True)
        data['colcands1_best_glob'] = cand_best

        return None

class MaskPredictorOracleLocalFlow(MaskPredictorIF):
    def __init__(self, config):
        super(MaskPredictorOracleLocalFlow, self).__init__()
        log_info(f"Using Oracle for Mask Prediction CNN (Local + Flow)")
    def _predict(self, data):
        one  = torch.ones_like(data['matched10_hard'])  
        zero = torch.zeros_like(data['matched10_hard'])        
        data['matched10_img_flow'] = zero
        data['matched10_img_glob'] = zero
        data['matched10_img_loc' ] = one
        data['matched10_img_none'] = zero
        data['matched10'] = data['matched10_img_flow']  # backward compatibility

        # Select Best Candidate by Oracle Distance from GT:        
        cands =  [data['colcands1_loc'][:,i] for i in range(data['colcands1_loc'].shape[1])] # [N,K,C,H,W]
        cands += [data['i0cw']]  # add warped flow candidate
        cands = torch.stack(cands,dim=-1)
        distance = cands-data['i1c'][...,None] # [N,K,C,H,W]
        distance_ab_l2 = torch.norm(distance[:,1:3], dim=1, p=2, keepdim=True) # [N,K,1,H,W]
        cand_best, _ = get_min_value_lastdim(cands, distance_ab_l2, get_best_cond_val=True)
        data['colcands1_best_loc'] = cand_best

        return None

class MaskPredictorOracleGlobalFlow(MaskPredictorIF):
    def __init__(self, config):
        super(MaskPredictorOracleGlobalFlow, self).__init__()
        log_info(f"Using Oracle for Mask Prediction CNN (Global + Flow)")
    def _predict(self, data):
        one  = torch.ones_like(data['matched10_hard'])  
        zero = torch.zeros_like(data['matched10_hard'])        
        data['matched10_img_flow'] = zero
        data['matched10_img_glob'] = one
        data['matched10_img_loc' ] = zero
        data['matched10_img_none'] = zero
        data['matched10'] = data['matched10_img_flow']  # backward compatibility

        # Select Best Candidate by Oracle Distance from GT:        
        cands =  [data['colcands1_glob'][:,i] for i in range(data['colcands1_glob'].shape[1])] # [N,K,C,H,W]
        cands += [data['i0cw']]  # add warped flow candidate
        cands = torch.stack(cands,dim=-1)
        distance = cands-data['i1c'][...,None] # [N,K,C,H,W]
        distance_ab_l2 = torch.norm(distance[:,1:3], dim=1, p=2, keepdim=True) # [N,K,1,H,W]
        cand_best, _ = get_min_value_lastdim(cands, distance_ab_l2, get_best_cond_val=True)
        data['colcands1_best_glob'] = cand_best

        return None

class MaskPredictorOracleGlobalLocalFlow(MaskPredictorIF):
    def __init__(self, config):
        super(MaskPredictorOracleGlobalLocalFlow, self).__init__()
        log_info(f"Using Oracle for Mask Prediction CNN (Local + Global + Flow)")
    def _predict(self, data):
        one  = torch.ones_like(data['matched10_hard'])  
        zero = torch.zeros_like(data['matched10_hard'])        
        data['matched10_img_flow'] = zero
        data['matched10_img_glob'] = one
        data['matched10_img_loc' ] = zero
        data['matched10_img_none'] = zero
        data['matched10'] = data['matched10_img_flow']  # backward compatibility

        # Select Best Candidate by Oracle Distance from GT:        
        cands =  [data['colcands1_glob'][:,i] for i in range(data['colcands1_glob'].shape[1])] # [N,K,C,H,W]
        cands +=  [data['colcands1_loc'][:,i] for i in range(data['colcands1_loc'].shape[1])] # [N,K,C,H,W]
        cands += [data['i0cw']]  # add warped flow candidate
        cands = torch.stack(cands,dim=-1)
        distance = cands-data['i1c'][...,None] # [N,K,C,H,W]
        distance_ab_l2 = torch.norm(distance[:,1:3], dim=1, p=2, keepdim=True) # [N,K,1,H,W]
        cand_best, _ = get_min_value_lastdim(cands, distance_ab_l2, get_best_cond_val=True)
        data['colcands1_best_glob'] = cand_best

        return None

# END ORACLE SECTION
##############################################################################################


class MaskPredictorKeepAllLocalCands(MaskPredictorIF):
    def __init__(self, config):
        super(MaskPredictorKeepAllLocalCands, self).__init__()
    def _predict(self, data):
        one  = torch.ones_like(data['matched10_hard'])  
        zero = torch.zeros_like(data['matched10_hard'])        
        data['matched10_img_flow'] = zero
        data['matched10_img_glob'] = zero
        data['matched10_img_loc' ] = one
        data['matched10_img_none'] = zero

        data['matched10'] = data['matched10_img_flow']  # backward compatibility
        return None

class MaskPredictorKeepAllGlobalCands(MaskPredictorIF):
    def __init__(self, config):
        super(MaskPredictorKeepAllGlobalCands, self).__init__()
    def _predict(self, data):
        one  = torch.ones_like(data['matched10_hard'])  
        zero = torch.zeros_like(data['matched10_hard'])        
        data['matched10_img_flow'] = zero
        data['matched10_img_glob'] = one
        data['matched10_img_loc' ] = zero
        data['matched10_img_none'] = zero

        data['matched10'] = data['matched10_img_flow']  # backward compatibility
        return None

class MaskPredictorKeepAllFlow(MaskPredictorIF):
    def __init__(self, config):
        super(MaskPredictorKeepAllFlow, self).__init__()
    def _predict(self, data):
        one  = torch.ones_like(data['matched10_hard'])  
        zero = torch.zeros_like(data['matched10_hard'])        
        data['matched10_img_flow'] = one
        data['matched10_img_glob'] = zero
        data['matched10_img_loc' ] = zero
        data['matched10_img_none'] = zero
        data['matched10'] = data['matched10_img_flow']  # backward compatibility
        return None



class MaskPredictorKeepAllGlobalCandsAndTDV(MaskPredictorIF):
    def __init__(self, config):
        super(MaskPredictorKeepAllGlobalCandsAndTDV, self).__init__()
    def _predict(self, data):
        one  = torch.ones_like(data['matched10_hard'])  
        zero = torch.zeros_like(data['matched10_hard'])        
        data['matched10_img_flow'] = data['matched10_dat_flow'] = zero.clone()
        data['matched10_img_glob'] = data['matched10_dat_glob'] = one.clone()
        data['matched10_img_loc' ] = data['matched10_dat_loc' ] = zero.clone()
        data['matched10_img_none'] = data['matched10_dat_none'] = zero.clone()
        data['lambda_tdv_mul']  =  one.clone()

        data['matched10'] = data['matched10_img_flow']  # backward compatibility

class MaskPredictorSoftHeuristic(MaskPredictorIF):
    def __init__(self, config):
        super(MaskPredictorSoftHeuristic, self).__init__()
    def _predict(self, data):
        data['matched10'] = data['matched10_soft']  # keep soft prediction

class MaskPredictorCnnVar(MaskPredictorIF):
    def __init__(self, config):
        super(MaskPredictorCnnVar, self).__init__()

        self._is_TORCHSCRIPT_EXPORT:bool = TORCHSCRIPT_EXPORT # Hack for Torchscript export

        assert 'no_head' in  config and config['no_head'], f"MaskPredictorCnnVar requires that the underlying CNN is a headless feature encoder "
        self.CNNPredictor = MaskPredictor(config)
        self.config = config
        self.num_features = config['num_features']
        self.bias_out = config['bias_out'] if 'bias_out' in config else False

        self.use_colcand_flow = False
        self.use_colcand_glob = False
        self.use_colcand_loc = False

        cands_allowed = ['flow', 'glob', 'loc']
        if not set(config['cands']).issubset(cands_allowed):
            raise ValueError(f"MaskPredctiorCNN: unknown settings detected in 'cands':{config['cands']} but must be in {cands_allowed}")
        self.mask_keys = config['cands'][:]
        if 'flow' in config['cands']:
            self.use_colcand_flow = True
            log_info(f"MaskCNN: Using use_colcand_flow:{self.use_colcand_flow }")   
        if 'glob' in config['cands']:
            self.use_colcand_glob = True
            log_info(f"MaskCNN: Using use_colcand_glob:{self.use_colcand_glob }")   
        if 'loc'  in config['cands']:
            self.use_colcand_loc  = True
            log_info(f"MaskCNN: Using use_colcand_loc:{self.use_colcand_loc }")   
        if self.use_colcand_flow == self.use_colcand_glob == self.use_colcand_loc == False:
            raise ValueError(f"Seems like no candidate inpute was specified! {config['cands']}")
                         

        self._use_warped_prev_matched10_img_input = False
        if 'use_warped_prev_matched10_img_input' in config and config['use_warped_prev_matched10_img_input']:
            self._use_warped_prev_matched10_img_input = True
            if 'mode' in config['use_warped_prev_mask_init_val'] and config['use_warped_prev_mask_init_val']['mode'] == 'learned':
                self._init_val__matched10_img = get_parameter_v2('init_val__matched10_img', config['use_warped_prev_mask_init_val'])
            else:
                self._init_val__matched10_img = config['use_warped_prev_mask_init_val']['init']
            log_info(f"MaskCNN: Using additional previous matched mask as input to MaskCNN 'use_warped_prev_matched10_img_input':{self._use_warped_prev_matched10_img_input}")   
        self._use_warped_prev_matched10_dat_input = False
        if 'use_warped_prev_matched10_dat_input' in config and config['use_warped_prev_matched10_dat_input']:
            self._use_warped_prev_matched10_dat_input = True
            self._init_val__matched10_dat = config['use_warped_prev_mask_init_val']['init']
            if 'mode' in config['use_warped_prev_mask_init_val'] and config['use_warped_prev_mask_init_val']['mode'] == 'learned':
                self._init_val__matched10_dat = get_parameter_v2('init_val__matched10_dat', config['use_warped_prev_mask_init_val'])
            else:
                self._init_val__matched10_dat = config['use_warped_prev_mask_init_val']['init']
            log_info(f"MaskCNN: Using additional previous matched mask as input to MaskCNN 'use_warped_prev_matched10_dat_input':{self._use_warped_prev_matched10_dat_input}")            

        #TODO: energy_est => energy_reg in Config (keep for now for backward compatibility)
        self._use_warped_prev_energy_reg_fin_input = False
        if 'use_warped_prev_energy_est_input' in config and config['use_warped_prev_energy_est_input']:
            self._use_warped_prev_energy_reg_fin_input = True        
            if 'mode' in config['use_warped_prev_energy_est_init_val'] and config['use_warped_prev_energy_est_init_val']['mode'] == 'learned':
                self._init_val__energy = get_parameter_v2('init_val__energy', config['use_warped_prev_energy_est_init_val'])
            else:
                self._init_val__energy = config['use_warped_prev_energy_est_init_val']['init']
            log_info(f"MaskCNN: Using additional previous energy as input to MaskCNN 'use_warped_prev_energy_est_input':{self._use_warped_prev_energy_reg_fin_input}")            

        self._use_warped_prev_lambda_tdv_mul = False
        if 'use_warped_prev_lambda_tdv_mul' in config and config['use_warped_prev_lambda_tdv_mul']:
            self._use_warped_prev_lambda_tdv_mul = True        
            if 'mode' in config['use_warped_prev_lambda_tdv_mul_init_val'] and config['use_warped_prev_lambda_tdv_mul_init_val']['mode'] == 'learned':
                self._init_val__lambda_tdv_mul = get_parameter_v2('init_val__lambda_tdv_mul',config['use_warped_prev_lambda_tdv_mul_init_val'])
            else:
                self._init_val__lambda_tdv_mul = config['use_warped_prev_lambda_tdv_mul_init_val']['init']
            log_info(f"MaskCNN: Using additional previous energy as input to MaskCNN 'use_warped_prev_lambda_tdv_mul':{self._use_warped_prev_lambda_tdv_mul}")            
        
        self._uses_warped_inputs=False
        if (self._use_warped_prev_matched10_img_input or  self._use_warped_prev_matched10_dat_input or
            self._use_warped_prev_energy_reg_fin_input or self._use_warped_prev_lambda_tdv_mul or self._use_OOIM_input):
            self._use_warped_inputs=True

        self._use_gray_input = False
        if 'use_gray_input' in config and config['use_gray_input']:
            self._use_gray_input = True
            log_info(f"Using additional Gray Value input to MaskCNN 'use_gray_input':{self._use_gray_input}")

        self._use_OOIM_input = False
        if 'use_OOIM' in config and config['use_OOIM']:
            self._use_OOIM_input = True
            log_info(f"Using additional OutOfImage mask as input to MaskCNN 'use_OOIM':{self._use_OOIM_input}")

        self._use_confidence_input = False
        if 'use_confidence_input' in config and config['use_confidence_input']:
            self._use_confidence_input = True
            log_info(f"MaskCNN: Using confidence data as input  {self._use_confidence_input}")

        self.allow_none_mask = False
        if 'allow_none_mask' in config and config['allow_none_mask']:
            self.allow_none_mask = True
            self.mask_keys += ['none']  # add a none key => 'flow', 'none'
            log_info(f"MaskCNN: allow_none_mask {self.allow_none_mask}")

        self.seperate_dataterm_mask = False
        if 'seperate_dataterm_mask' in config and config['seperate_dataterm_mask']:
            self.seperate_dataterm_mask = True
            log_info(f"MaskCNN: Generating a seperate masks for Init_image and Dataterm {self.seperate_dataterm_mask}")     

        self.lambda_tdv_mul = False
        if 'lambda_tdv_mul' in config and config['lambda_tdv_mul']:
            self.lambda_tdv_mul = True
            log_info(f"MaskCNN: Using lambda_tdv_mul {self.lambda_tdv_mul}")

        # Add output head for init image masks
        self._img_masks_out = []
        if self.use_colcand_flow:
            self._img_masks_out += [ 'matched10_img_flow']
        if self.use_colcand_glob:
            self._img_masks_out += [ 'matched10_img_glob']
        if self.use_colcand_loc:
            self._img_masks_out += [ 'matched10_img_loc']
        if self.allow_none_mask:
            self._img_masks_out += [ 'matched10_img_none']   
        self._head_img_masks = Conv2d(self.num_features, len(self._img_masks_out), 1, invariant=False, bound_norm=False, bias=self.bias_out) 

        # Add output head for dataterm image masks
        self._dat_masks_out = []
        if self.seperate_dataterm_mask:
            if self.use_colcand_flow:
                self._dat_masks_out += [ 'matched10_dat_flow']
            if self.use_colcand_glob:
                self._dat_masks_out += [ 'matched10_dat_glob']
            if self.use_colcand_loc:
                self._dat_masks_out += [ 'matched10_dat_loc']
            if self.allow_none_mask:
                self._dat_masks_out += [ 'matched10_dat_none']   
            self._head_dat_masks = Conv2d(self.num_features, len(self._dat_masks_out), 1, invariant=False, bound_norm=False, bias=self.bias_out) 

        # Add output head for Regularization mask
        if self.lambda_tdv_mul:
            self._head_lambda_tdv_mul = Conv2d(self.num_features, 1, 1, invariant=False, bound_norm=False, bias=self.bias_out) 

        self.sigmoid = torch.sigmoid

        self._use_all_mask_inputs_legacy = False
        if 'use_all_mask_inputs_legacy' in config and config['use_all_mask_inputs_legacy']:
            self.__use_all_mask_inputs_legacy = True
            self.mask_keys = ['flow', 'glob', 'loc', 'none']  # add 3x2 layers to simulate flow input
            log_info(f"Using  MaskCNN with identical amount of input layers :{self.__use_all_mask_inputs_legacy}")


    def set_lr_mul(self, lr_mul):
        for name, param in  self.named_parameters():
            lr_mul_old = param._lr_mul if hasattr(param, "_lr_mul" ) else  1
            param._lr_mul = lr_mul * lr_mul_old
            log_info(f" {lr_mul_old:.2e}->{param._lr_mul:.2e} : {name}")

    def _predict(self, data: Dict[str, torch.Tensor]) -> Optional[List[torch.Tensor]]:
        if self._use_OOIM_input and ('flow10' not in data):
            raise RuntimeError(f"'flow10' not found in data, but needed for OOIM computation!")
        if self._uses_warped_inputs and  ('flow10' not in data):
            warnings.warn(f"No Optical Flow is present but the MaskCNN uses inputs from the previous run - these inputs are not Warped!")

        # Changing order here breaks backward compatibility! (new version is not direct compatible with old one!)
        cnn_inputs = []
        if self.use_colcand_flow:
            df_cand_flow = data['fdelta10']

            dl_cand_flow = (data['i0cw'][:, 0:1] - data['i1L'][:,0:1]).abs()
            ab_cand_flow =  data['i0cw'][:, 1:3]
            cnn_inputs += [ df_cand_flow, dl_cand_flow , ab_cand_flow ] 
        
        if self.use_colcand_glob:
            colcand_glob = data[f"colcands1_best_glob"]
            dl_cand_glob = (colcand_glob[:, 0:1]- data['i1L'][:,0:1]).abs()
            ab_cand_glob =  colcand_glob[:, 1:3]
            cnn_inputs += [ dl_cand_glob, ab_cand_glob]
            if self._use_confidence_input:
                if not 'conf_best_glob' in data: raise ValueError("conf_best_glob is missing => check if config['data']['colcands_conf_best'] is set to True!")
                cnn_inputs += [ data['conf_best_glob'] ]

        if self.use_colcand_loc:
            colcand_loc = data[f"colcands1_best_loc"]
            dl_cand3 = (colcand_loc[:,0:1]- data['i1L'][:,0:1]).abs()
            ab_cand3 =  colcand_loc[:,1:3]
            cnn_inputs += [dl_cand3, ab_cand3]
            if self._use_confidence_input:
                if not 'conf_best_loc' in data: raise ValueError("conf_best_loc is missing => check if config['data']['colcands_conf_best'] is set to True!")
                cnn_inputs += [data[f"conf_best_loc"]]  

        if self._use_gray_input:
            cnn_inputs = [ data['i1L'][:,0:1], ] + cnn_inputs
        if self._use_OOIM_input:
            nOOIM_mask = flow_warp( torch.ones_like(data['i1L'][:,0:1]), data['flow10'][:,0:2])
            cnn_inputs += [ nOOIM_mask ]

        if self._use_warped_prev_matched10_img_input:
            for key in self.mask_keys:  #  ['flow', 'glob', 'loc' ,'none']
                mask_key = f"matched10_img_{key}_prev_warp"
                if mask_key in data: 
                    cnn_inputs += [ data[mask_key] ]       
                else:
                    cnn_inputs += [ self._init_val__matched10_img * torch.ones_like(data['i1L'][:,0:1]) ]
        if self._use_warped_prev_matched10_dat_input:
            for key in self.mask_keys:  #  ['flow', 'glob', 'loc' ,'none']
                mask_key = f"matched10_dat_{key}_prev_warp"
                if mask_key in data: 
                    cnn_inputs += [ data[mask_key] ]       
                else:
                    cnn_inputs += [ self._init_val__matched10_dat * torch.ones_like(data['i1L'][:,0:1]) ]
        
        if self._use_warped_prev_energy_reg_fin_input:
            if "energy_reg_fin_prev_warp" in data: 
                cnn_inputs += [ data["energy_reg_fin_prev_warp"] ]       
            else:
                cnn_inputs += [  self._init_val__energy * torch.ones_like(data['i1L'][:,0:1]) ]

        if self._use_warped_prev_lambda_tdv_mul:
            if "lambda_tdv_mul_prev_warp" in data: 
                cnn_inputs += [ data["lambda_tdv_mul_prev_warp"] ]       
            else:
                cnn_inputs += [  self._init_val__lambda_tdv_mul * torch.ones_like(data['i1L'][:,0:1]) ]

        cnn_inputs = torch.cat(cnn_inputs, dim=1)

        feats, _ = self.CNNPredictor(cnn_inputs)            

        # Compute Outputs for init image masks
        img_masks = self._head_img_masks(feats)
        img_masks = F.softmax(img_masks, dim=1)
        for i, mask_name in enumerate(self._img_masks_out):
            data[mask_name] = img_masks[:,i:i+1]
        
        if self.seperate_dataterm_mask:
            # Compute Outputs for dataterm image masks
            dat_masks = self._head_dat_masks(feats)
            dat_masks = F.softmax(dat_masks, dim=1)
            for i, mask_name in enumerate(self._dat_masks_out):
                data[mask_name] = dat_masks[:,i:i+1]
        else:  # not seperate => use _img_ masks (init image generation) for _dat_ (dataterm)
            for i, img_mask_name in enumerate(self._img_masks_out):
                dat_mask_name = img_mask_name.replace("_img_","_dat_")
                data[dat_mask_name] = data[img_mask_name] 
        
        if self.lambda_tdv_mul:
            lambda_tdv_mul = self._head_lambda_tdv_mul(feats)
            data['lambda_tdv_mul']  = self.sigmoid(lambda_tdv_mul) 

        if 'matched10_img_flow' in data:
            data['matched10'] = data['matched10_img_flow']  # backward compatibility
        else:
            warnings.warn(f"\n NoFlow Mode not yet fully tested \n")
            data['matched10'] =  torch.zeros_like(data['i1L'][:,0:1])  # backward compatibility
            data['matched10_img_flow'] =  torch.zeros_like(data['i1L'][:,0:1])  # backward compatibility
        return None



class MaskPredictorCnnVarCmbdHead(MaskPredictorIF):
    def __init__(self, config):
        super(MaskPredictorCnnVarCmbdHead, self).__init__()
        assert 'no_head' in  config and config['no_head'], f"MaskPredictorCnnVar requires that the underlying CNN is a headless feature encoder "
        self.CNNPredictor = MaskPredictor(config)
        self.config = config
        self.num_features = config['num_features']
        self.bias_out = config['bias_out']  # first run of refactor was True

        self.use_colcand_flow = False
        self.use_colcand_glob = False
        self.use_colcand_loc = False

        cands_allowed = ['flow', 'glob', 'loc']
        if not set(config['cands']).issubset(cands_allowed):
            raise ValueError(f"MaskPredctiorCNN: unknown settings detected in 'cands':{config['cands']} but must be in {cands_allowed}")
        self.mask_keys = config['cands'][:]
        if 'flow' in config['cands']:
            self.use_colcand_flow = True
            log_info(f"MaskCNN: Using use_colcand_flow:{self.use_colcand_flow }")   
        if 'glob' in config['cands']:
            self.use_colcand_glob = True
            log_info(f"MaskCNN: Using use_colcand_glob:{self.use_colcand_glob }")   
        if 'loc'  in config['cands']:
            self.use_colcand_loc  = True
            log_info(f"MaskCNN: Using use_colcand_loc:{self.use_colcand_loc }")   
        if self.use_colcand_flow == self.use_colcand_glob == self.use_colcand_loc == False:
            raise ValueError(f"Seems like no candidate inpute was specified! {config['cands']}")
        
        self._use_warped_prev_matched10_img_input = False
        if 'use_warped_prev_matched10_img_input' in config and config['use_warped_prev_matched10_img_input']:
            self._use_warped_prev_matched10_img_input = True
            if 'mode' in config['use_warped_prev_mask_init_val'] and config['use_warped_prev_mask_init_val']['mode'] == 'learned':
                self._init_val__matched10_img = get_parameter_v2('init_val__matched10_img', config['use_warped_prev_mask_init_val'])
            else:
                self._init_val__matched10_img = config['use_warped_prev_mask_init_val']['init']
            log_info(f"MaskCNN: Using additional previous matched mask as input to MaskCNN 'use_warped_prev_matched10_img_input':{self._use_warped_prev_matched10_img_input}")   
        self._use_warped_prev_matched10_dat_input = False
        if 'use_warped_prev_matched10_dat_input' in config and config['use_warped_prev_matched10_dat_input']:
            self._use_warped_prev_matched10_dat_input = True
            self._init_val__matched10_dat = config['use_warped_prev_mask_init_val']['init']
            if 'mode' in config['use_warped_prev_mask_init_val'] and config['use_warped_prev_mask_init_val']['mode'] == 'learned':
                self._init_val__matched10_dat = get_parameter_v2('init_val__matched10_dat', config['use_warped_prev_mask_init_val'])
            else:
                self._init_val__matched10_dat = config['use_warped_prev_mask_init_val']['init']
            log_info(f"MaskCNN: Using additional previous matched mask as input to MaskCNN 'use_warped_prev_matched10_dat_input':{self._use_warped_prev_matched10_dat_input}")            

        #TODO: energy_est => energy_reg in Config (keep for now for backward compatibility)
        self._use_warped_prev_energy_reg_fin_input = False
        if 'use_warped_prev_energy_est_input' in config and config['use_warped_prev_energy_est_input']:
            self._use_warped_prev_energy_reg_fin_input = True        
            if 'mode' in config['use_warped_prev_energy_est_init_val'] and config['use_warped_prev_energy_est_init_val']['mode'] == 'learned':
                self._init_val__energy = get_parameter_v2('init_val__energy', config['use_warped_prev_energy_est_init_val'])
            else:
                self._init_val__energy = config['use_warped_prev_energy_est_init_val']['init']
            log_info(f"MaskCNN: Using additional previous energy as input to MaskCNN 'use_warped_prev_energy_est_input':{self._use_warped_prev_energy_reg_fin_input}")            

        self._use_warped_prev_lambda_tdv_mul = False
        if 'use_warped_prev_lambda_tdv_mul' in config and config['use_warped_prev_lambda_tdv_mul']:
            self._use_warped_prev_lambda_tdv_mul = True        
            if 'mode' in config['use_warped_prev_lambda_tdv_mul_init_val'] and config['use_warped_prev_lambda_tdv_mul_init_val']['mode'] == 'learned':
                self._init_val__lambda_tdv_mul = get_parameter_v2('init_val__lambda_tdv_mul',config['use_warped_prev_lambda_tdv_mul_init_val'])
            else:
                self._init_val__lambda_tdv_mul = config['use_warped_prev_lambda_tdv_mul_init_val']['init']
            log_info(f"MaskCNN: Using additional previous energy as input to MaskCNN 'use_warped_prev_lambda_tdv_mul':{self._use_warped_prev_lambda_tdv_mul}")            
        
        self._uses_warped_inputs=False
        if (self._use_warped_prev_matched10_img_input or  self._use_warped_prev_matched10_dat_input or
            self._use_warped_prev_energy_reg_fin_input or self._use_warped_prev_lambda_tdv_mul or self._use_OOIM_input):
            self._use_warped_inputs=True

        self._use_gray_input = False
        if 'use_gray_input' in config and config['use_gray_input']:
            self._use_gray_input = True
            log_info(f"Using additional Gray Value input to MaskCNN 'use_gray_input':{self._use_gray_input}")

        self._use_OOIM_input = False
        if 'use_OOIM' in config and config['use_OOIM']:
            self._use_OOIM_input = True
            log_info(f"Using additional OutOfImage mask as input to MaskCNN 'use_OOIM':{self._use_OOIM_input}")

        self._use_confidence_input = False
        if 'use_confidence_input' in config and config['use_confidence_input']:
            self._use_confidence_input = True
            log_info(f"MaskCNN: Using confidence data as input  {self._use_confidence_input}")

        self.allow_none_mask = False
        if 'allow_none_mask' in config and config['allow_none_mask']:
            self.allow_none_mask = True
            self.mask_keys += ['none']  # add a none key => 'flow', 'none'
            log_info(f"MaskCNN: allow_none_mask {self.allow_none_mask}")

        self.seperate_dataterm_mask = False
        if 'seperate_dataterm_mask' in config and config['seperate_dataterm_mask']:
            self.seperate_dataterm_mask = True
            log_info(f"MaskCNN: Generating a seperate masks for Init_image and Dataterm {self.seperate_dataterm_mask}")     

        self.lambda_tdv_mul = False
        if 'lambda_tdv_mul' in config and config['lambda_tdv_mul']:
            self.lambda_tdv_mul = True
            log_info(f"MaskCNN: Using lambda_tdv_mul {self.lambda_tdv_mul}")

        # Add output head for init image masks
        self._img_masks_out = []
        if self.use_colcand_flow:
            self._img_masks_out += [ 'matched10_img_flow']
        if self.use_colcand_glob:
            self._img_masks_out += [ 'matched10_img_glob']
        if self.use_colcand_loc:
            self._img_masks_out += [ 'matched10_img_loc']
        if self.allow_none_mask:
            self._img_masks_out += [ 'matched10_img_none']   
        self._out_feats =  [ self._img_masks_out ]
        # self._head_img_masks = Conv2d(self.num_features, len(self._img_masks_out), 1, invariant=False, bound_norm=False, bias=self.bias_out) 

        # Add output head for dataterm image masks
        self._dat_masks_out = []
        if self.seperate_dataterm_mask:
            if self.use_colcand_flow:
                self._dat_masks_out += [ 'matched10_dat_flow']
            if self.use_colcand_glob:
                self._dat_masks_out += [ 'matched10_dat_glob']
            if self.use_colcand_loc:
                self._dat_masks_out += [ 'matched10_dat_loc']
            if self.allow_none_mask:
                self._dat_masks_out += [ 'matched10_dat_none']   

            self._out_feats +=  [ self._dat_masks_out ]
            # self._head_dat_masks = Conv2d(self.num_features, len(self._dat_masks_out), 1, invariant=False, bound_norm=False, bias=self.bias_out) 

        # Add output head for Regularization mask
        if self.lambda_tdv_mul:
            self._out_feats +=  [[ 'lambda_tdv_mul' ]]

        # Combined feature count
        self._feat_cnt = sum( [len(f) for f in self._out_feats] )
        self._head_cmbd = Conv2d(self.num_features, self._feat_cnt, 1, invariant=False, bound_norm=False, bias=self.bias_out) 

        self.sigmoid = torch.sigmoid

    def set_lr_mul(self, lr_mul):
        for name, param in  self.named_parameters():
            lr_mul_old = param._lr_mul if hasattr(param, "_lr_mul" ) else  1
            param._lr_mul = lr_mul * lr_mul_old
            log_info(f" {lr_mul_old:.2e}->{param._lr_mul:.2e} : {name}")

    def _predict(self, data):
        if self._use_OOIM_input and ('flow10' not in data):
            raise RuntimeError(f"'flow10' not found in data, but needed for OOIM computation!")
        if self._uses_warped_inputs and  ('flow10' not in data):
            warnings.warn(f"No Optical Flow is present but the MaskCNN uses inputs from the previous run - these inputs are not Warped!")

        # Changing order here breaks backward compatibility! (new version is not direct compatible with old one!)
        cnn_inputs = []
        if self.use_colcand_flow:
            df_cand_flow = data['fdelta10']
            dl_cand_flow = (data['i0cw'][:, 0:1] - data['i1L'][:,0:1]).abs()
            ab_cand_flow =  data['i0cw'][:, 1:3]
            cnn_inputs += [ df_cand_flow, dl_cand_flow , ab_cand_flow ] 
        
        if self.use_colcand_glob:
            colcand_glob = data[f"colcands1_best_glob"]
            dl_cand_glob = (colcand_glob[:, 0:1]- data['i1L'][:,0:1]).abs()
            ab_cand_glob =  colcand_glob[:, 1:3]
            cnn_inputs += [ dl_cand_glob, ab_cand_glob]
            if self._use_confidence_input:
                if not 'conf_best_glob' in data: raise ValueError("conf_best_glob is missing => check if config['data']['colcands_conf_best'] is set to True!")
                cnn_inputs += [ data['conf_best_glob'] ]

        if self.use_colcand_loc:
            colcand_loc = data[f"colcands1_best_loc"]
            dl_cand3 = (colcand_loc[:,0:1]- data['i1L'][:,0:1]).abs()
            ab_cand3 =  colcand_loc[:,1:3]
            cnn_inputs += [dl_cand3, ab_cand3]
            if self._use_confidence_input:
                if not 'conf_best_loc' in data: raise ValueError("conf_best_loc is missing => check if config['data']['colcands_conf_best'] is set to True!")
                cnn_inputs += [data[f"conf_best_loc"]]  

        if self._use_gray_input:
            cnn_inputs = [ data['i1L'][:,0:1], ] + cnn_inputs
        if self._use_OOIM_input:
            nOOIM_mask = flow_warp( torch.ones_like(data['i1L'][:,0:1]), data['flow10'][:,0:2])
            cnn_inputs += [ nOOIM_mask ]

        if self._use_warped_prev_matched10_img_input:
            for key in self.mask_keys:  #  ['flow', 'glob', 'loc' ,'none']
                mask_key = f"matched10_img_{key}_prev_warp"
                if mask_key in data: 
                    cnn_inputs += [ data[mask_key] ]       
                else:
                    cnn_inputs += [ self._init_val__matched10_img * torch.ones_like(data['i1L'][:,0:1]) ]
        if self._use_warped_prev_matched10_dat_input:
            for key in self.mask_keys:  #  ['flow', 'glob', 'loc' ,'none']
                mask_key = f"matched10_dat_{key}_prev_warp"
                if mask_key in data: 
                    cnn_inputs += [ data[mask_key] ]       
                else:
                    cnn_inputs += [ self._init_val__matched10_dat * torch.ones_like(data['i1L'][:,0:1]) ]
        
        if self._use_warped_prev_energy_reg_fin_input:
            if "energy_reg_fin_prev_warp" in data: 
                cnn_inputs += [ data["energy_reg_fin_prev_warp"] ]       
            else:
                cnn_inputs += [  self._init_val__energy * torch.ones_like(data['i1L'][:,0:1]) ]

        if self._use_warped_prev_lambda_tdv_mul:
            if "lambda_tdv_mul_prev_warp" in data: 
                cnn_inputs += [ data["lambda_tdv_mul_prev_warp"] ]       
            else:
                cnn_inputs += [  self._init_val__lambda_tdv_mul * torch.ones_like(data['i1L'][:,0:1]) ]

        cnn_inputs = torch.cat(cnn_inputs, dim=1)
        feats, _ = self.CNNPredictor(cnn_inputs)


        # Compute output masks from output features
        idx = 0
        out_feats = self._head_cmbd(feats)
        for names in self._out_feats:
            feat_cnt = len(names)
            c_feats = out_feats[:,idx: idx+feat_cnt]
            if feat_cnt == 1:
                c_feats = torch.sigmoid(c_feats)
            else:
                c_feats = F.softmax(c_feats, dim=1)
            for i, name in enumerate(names):
                data[name] = c_feats[:,i:i+1]
                log_info(f"MaskCNN - feat outputs: {idx+i} {idx+i+1} {name}  {c_feats[:,i:i+1].shape}")
            idx +=i+1

        # If not using dataterm masks => duplicate init image masks
        if not self.seperate_dataterm_mask:
            for i, img_mask_name in enumerate(self._img_masks_out):
                dat_mask_name = img_mask_name.replace("_img_","_dat_")
                data[dat_mask_name] = data[img_mask_name] 

        data['matched10'] = data['matched10_img_flow']  # backward compatibility
        return None


