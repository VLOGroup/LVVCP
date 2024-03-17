import PIL
import numpy as np
import torch
import imageio
from torchvision import transforms
from scipy.ndimage import gaussian_filter
from skimage.morphology import diamond, erosion, dilation
from common_utils.color_conversion_pytorch import rgb2lab_normalized_NCHW, lab2rgb_normalized_NCHW
from common_utils.flow_utils import flow_warp, flow_warp_with_mask
import torchvision.transforms.functional as TF

from typing import Dict, List, Tuple, Optional

# Required for RandomFlipTH, RandomRotateTH, RandomResizedCropTH to work with flow vectors which are floating point
maj_verison, min_verison = [int(v) for v in torch.__version__.split(".")[0:2]]
if maj_verison < 2:
    assert min_verison >= 7,  f"Pytorch Version needs to be >= 1.7 for this dataloader to work. It uses the new functionality to augment tensor objects not only PIL objects. Version found={torch.__version__}"

def propagate_old_masks(data: Dict[str, torch.Tensor], data_prev: Dict[str, torch.Tensor], matched10_est:Optional[torch.Tensor]=None):
    """ Propagates the old masks to the new frame - if flow is present the masks will be warped"""
    for key in ['matched10_img_flow', 'matched10_img_glob', 'matched10_img_loc',  'matched10_img_none', 
                'matched10_dat_flow',  'matched10_dat_glob','matched10_dat_loc', 'matched10_dat_none',
                'energy_reg_fin', 'lambda_tdv_mul']:
        if (data_prev is not None) and key in data_prev:
            data[f"{key}_prev"     ] = data_prev[key] 
            if 'flow10' in data:
                data[f"{key}_prev_warp"] = flow_warp(data_prev[key] , data['flow10']) # Masks have 0 at 0, imgs have 0 at 0.5, hence  mask_warp * mask is ok here, but not for color
            else:
                data[f"{key}_prev_warp"] = data_prev[key] 
    
    if matched10_est is not None: # Previous estimate
        data['matched10_prev'     ] =  matched10_est  # may be used by MaskCNN as input
        if 'flow10' in data:
            data['matched10_prev_warp'] = flow_warp(matched10_est, data['flow10']) # Masks have 0 at 0, imgs have 0 at 0.5, hence  mask_warp * mask is ok here, but not for color
        else:
            data['matched10_prev_warp'] = matched10_est
    return data

def drop_prev_masks(data: Dict[str, torch.Tensor]):
    """remove masks from previous frames - just to make sure nothing is double """
    for key in ['matched10_img_flow', 'matched10_img_glob', 'matched10_img_loc',  'matched10_img_none', 
                'matched10_dat_flow',  'matched10_dat_glob','matched10_dat_loc', 'matched10_dat_none',
                'energy_reg_fin', 'lambda_tdv_mul', 'matched10']:
        if f"{key}_prev_warp" in data:
            del data[f"{key}_prev_warp"]
    return data

def drop_dataterm_masks(data: Dict[str, torch.Tensor]):
    """removed dataterm mask and previous versions - just to make sure nothing is double """
    for key in ['matched10_dat_flow',  'matched10_dat_glob','matched10_dat_loc', 'matched10_dat_none',
                'energy_reg_fin', 'lambda_tdv_mul', 'matched10']:
        if f"{key}_prev_warp" in data:
            del data[f"{key}_prev_warp"]
    return data
