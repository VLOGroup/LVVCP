import torch
import numpy as np
import imageio
from skimage.color import rgb2lab, lab2rgb

def show_Lab_full_swing_for_RGB_double():
    # https://stackoverflow.com/questions/59769586/what-range-does-skimage-use-in-lab-color-space-for-each-channel
    # pre-compute maximum ranges for ab space:
    all_rgb_values = np.mgrid[0:256, 0:256, 0:256].astype(np.float64).transpose(1,2,3,0) # [256,256,256,3]
    all_rgb_values = all_rgb_values.reshape(256*256,256,3)/255.0
    all_std_lab_values = rgb2lab( all_rgb_values)
    Lab_max = np.max(all_std_lab_values,axis=(0,1))
    Lab_min = np.min(all_std_lab_values,axis=(0,1))
    Lab_full_swing = Lab_max - Lab_min
    print(f"Lab max: {Lab_max}")
    print(f"Lab min: {Lab_min}")
    print(f"Lab full swing: {Lab_full_swing}")
    # Lab max: [100.          98.23305386  94.47812228]
    # Lab min: [   0.        -86.18302974 -107.85730021]
    # Lab full swing: [100.   184.41608361 202.33542248]



def show_Lab_full_swing_for_RGB_float_ext():
    # Extended Range (sampling in between std uint8 points) - takes ~100GB RAM to compute and quite some time to run

    # https://stackoverflow.com/questions/59769586/what-range-does-skimage-use-in-lab-color-space-for-each-channel
    # pre-compute maximum ranges for ab space:
    all_rgb_values = np.mgrid[0:1024, 0:1024, 0:1024].astype(np.float32).transpose(1,2,3,0) # [1024,1024,1024,3]
    all_rgb_values = all_rgb_values.reshape(1024*1024,1024,3)/1023.0
    all_std_lab_values = rgb2lab( all_rgb_values)
    Lab_max = np.max(all_std_lab_values,axis=(0,1))
    Lab_min = np.min(all_std_lab_values,axis=(0,1))
    Lab_full_swing = Lab_max - Lab_min
    print(f"Lab max: {Lab_max}")
    print(f"Lab min: {Lab_min}")
    print(f"Lab full swing: {Lab_full_swing}")
    # show_Lab_full_swing_for_RGB_float_ext()              
    # Lab max: [100.           98.23305386   94.47812228 ]
    # Lab min: [   0.          -86.18302974 -107.85730021]
    # Lab full swing: [100.    184.41608361  202.33542248]


def get_ab_max_full_swing():
    """ Approximated maximum swing for the ab channels of the CIE-Lab color space
        max_{x \in CIE-Lab} {x_a x_b} - min_{x \in CIE-Lab} {x_a x_b} 
    """
    return 202.33542248


def compute_PSNRab_np( x, y_gt, pp_max=get_ab_max_full_swing()):
    """ Computes the PSNR of the ab color channels.
        
        Note that the CIE-Lab space is asymmetric.
        The maximum size for the 2 channels of the ab subspace is approximately 202.3354... 
    """
    assert len(x.shape) == 3,    f"Expecting data of the size HW2 but found {x.shape}; This should be a,b channels of CIE-Lab Space"
    assert len(y_gt.shape) == 3, f"Expecting data of the size HW2 but found {y_gt.shape}; This should be a,b channels of CIE-Lab Space"
    assert x.shape==y_gt.shape,  f"Expecting data to have identical shape but found {y_gt.shape} != {x.shape}"
    
    H,W,C = x.shape
    assert C == 2, f"This function assumes that both x & y are both the ab channels of the CIE-Lab Space"

    MSE = np.sum(  (x - y_gt )**2 )  / (H*W*C) # C=2, two channels
    MSE  = np.clip(MSE , 1e-12, np.inf)

    PSNR_ab     = 10 * np.log10(pp_max**2)      - 10 * np.log10( MSE )

    return PSNR_ab

def compute_PSNRab_th(x, y_gt, pp_max=get_ab_max_full_swing()):
    """ Computes the PSNR of the ab color channels.
        
        Note that the CIE-Lab space is asymmetric.
        The maximum size for the 2 channels of the ab subspace is approximately 202.3354... 
    """
    assert len(x.shape) == 4,    f"Expecting data of the size HW2 but found {x.shape}; This should be a,b channels of CIE-Lab Space"
    assert len(y_gt.shape) == 4, f"Expecting data of the size HW2 but found {y_gt.shape}; This should be a,b channels of CIE-Lab Space"
    assert x.shape==y_gt.shape,  f"Expecting data to have identical shape but found {y_gt.shape} != {x.shape}"
    
    N,C,H,W = x.shape
    assert C == 2, f"This function assumes that both x & y are both the ab channels of the CIE-Lab Space, with C=2 but C={C}"

    MSE =  ( (x - y_gt )**2 ).sum()  / (N*H*W*C) # C=2, two channels

    PSNR_ab     = 10 * np.log10(pp_max**2)      - 10 * torch.log10( MSE )

    return PSNR_ab

