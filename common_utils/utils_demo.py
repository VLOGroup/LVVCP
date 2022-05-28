
import torch
import imageio
import numpy as np
import os

from common_utils.color_conversion_pytorch import rgb2lab_normalized_NCHW, rgb2gray_viaLab_NCHW_uint8_th
from common_utils.utils import log_info
from RAFT_custom.core.utils.utils import InputPadder

def load_and_clean_fielname_txt(filenamestr:str):
    """ This function loads and cleans a txt file containing image filenames
        See filnamestr_to_refs for the expected input
    """

    if not os.path.isfile(filenamestr): raise ValueError(f"FileList {filenamestr} not found!")
    with open(filenamestr) as fp:
        file_list = [line.strip() for line in fp.readlines()]
    return file_list

def filnamestr_to_refs(filenamestr:str):
    """ This function receives a line of filenames "ref1.png|ref2.png|ref3.png target.png"
        and converts them to cleaned list of filenames
        refs = ["ref1", "ref2",..]
        tar = "target
        
        The convention is as follows
            at least 1-ref and 1-target seperated by 1 space character
            if multiple references are given, they are seperated by a pipe symbol |
    """
    filenamestr = filenamestr.strip() # remove potential line endings
    fns = filenamestr.split(" ")
    if len(fns) != 2:
        raise ValueError(f"Wrong input format for input list! Must be of format 'ref1|ref2 target' but found\n {filenamestr}")
    fn_refs = fns[0].split('|')
    fn_curr = fns[1]
    return fn_refs, fn_curr

def get_coclands_postfix(candbest_type):
    """Get Postfix for colcands1_best_c coclands1_best_cdl colcands1_best_conf  """
    return {'conf':'_c', 'confgray':'_cdl', 'gray':'_dl', 'drop':'_glob', 'loc':'_loc', 'glob':'_glob', 'MaskCNN_colpred':'_MaskCNN_colpred'}[candbest_type]


def get_data_for_next_run(data):
    """ extracts data needed for next run"""
    data_keep = {}
    for key in ['i1g3', 'matched10_hard', 'matched10_soft', 'matched10_img_flow', 'matched10_img_glob', 'matched10_img_loc', 'matched10_img_none', 'matched10_dat_flow', 'matched10_dat_glob', 'matched10_dat_loc', 'matched10_dat_none', 'lambda_tdv_mul', 'matched10']:
        data_keep[key] = data[key]
    return data_keep

def load_img(fn, device, dtype, padder=None, padto=16):
    """
    Load color or gray images and always return both
    color & gray versions as Lab=[N3HW] and gray=[N3HW]
    Alpha masks are either None, or 1-channel images

    """
    img_uint8 = imageio.imread(fn).astype(np.uint8)
    img_uint8 = torch.from_numpy(img_uint8)[None,...].to(device=device,non_blocking=True)
    img_float = img_uint8.to(dtype=dtype)/255.
    img_mask = None

    if len(img_uint8.shape) == 3: # [NHW]
        # Image is a true grayscale image
        img_gray3 = torch.stack( [img_float,img_float,img_float], dim=1)
        img_col_lab = rgb2lab_normalized_NCHW(img_gray3.clone())
    elif len(img_uint8.shape) == 4 and img_uint8.shape[-1] in [3,4]:  # [NHWC]
        # Image is a color image
        if img_uint8.shape[-1] == 4:
            img_mask = (img_uint8[:,None,...,3]//255).float()
        img_uint8 = img_uint8[...,0:3] # drop alpha channels
        img_float = img_float[...,0:3] # drop alpha channels
        # Image is a colour image => simulate grayscale 
        img_float = img_float.permute(0,3,1,2) # [NHWC] => [NCHW]
        img_col_lab = rgb2lab_normalized_NCHW(img_float)                               #RGB => [Lab]
        # simulate gray conversion including uint8, to be identical as if the file were saved and reloaded
        img_uint8 = img_uint8.permute(0,3,1,2) # [NHWC] => [NCHW]
        img_gray3 = rgb2gray_viaLab_NCHW_uint8_th(img_uint8).to(dtype=dtype)/255.0   #RGB => [L,L,L]
    else:
        raise ValueError(f"{img_uint8.shape}, {fn}")
    
    if padder is None:
        padder = InputPadder(img_col_lab.shape, ds=padto)
    else:
        if ( (padder.ht != img_col_lab.shape[-2]) or (padder.wd != img_col_lab.shape[-1]) ):
            raise ValueError(f"Reuising Padder requires shapes to be equal but are old:{padder.ht, padder.wd}, new:{img_col_lab.shape[-2:]}")
    img_col_lab = padder.pad(img_col_lab) [0]
    img_gray3   = padder.pad(img_gray3) [0]
    img_mask = None if img_mask is None else padder.pad(img_mask)[0]
    return img_col_lab, img_gray3, img_mask, padder



class CandOptionsParser():
    def __init__(self, config):
        self.useCandsFlow = False         # Default setup always uses Flow
        self.useCandsMatchGlobal = False  #
        self.useCandsMatchLocal = False   #

        settings2attribs = {'flow':'useCandsFlow', 'glob':'useCandsMatchGlobal', 'loc':'useCandsMatchLocal'}

        if ('cands' in config['MaskPredictor']['config']) or (type(config['D']['config']['init_image']) == list):
            log_info(f"Detected New Candidate specification ['MaskPredictor']['config']['cands']  & ['D']['config']['init_image']")
            for key, var in settings2attribs.items():
                if  key  in config['MaskPredictor']['config']['cands']:
                    setattr(self, var, True)
                    log_info(f"Found {key} in config['MaskPredictor']['config']['cands'] => turn on matching for {key}={getattr(self, var)}")
        else:
            raise ValueError(f"Old configuration way 'candbest' and 'candbest3' is not available any more ")

