# This file contains conversion between color spaces in PyTorch
# This implementation aims to yield similar values than SK_images color conversion utilities
# Unittests are implemented that verify that values do match within certain error margins
#

import torch
import numpy as np
from scipy import misc

from skimage.color import rgb2lab as sk_rgb2lab
from skimage.color import lab2rgb as sk_lab2rgb
from skimage.color import rgb2xyz as sk_rgb2xyz
from skimage.color import xyz2rgb as sk_xyz2rgb
from skimage.color import xyz2lab as sk_xyz2lab
from skimage.color import lab2xyz as sk_lab2xyz

def rgb2gray_viaLab_NCHW_uint8_th(img_rgb, keepC3=True):
    """ This Gray conversion simulates saveing to uint8 files"""
    if img_rgb.max() < 1.: raise ValueError(f"maximum value of image is  < 1, should be 0..255")
    return (_rgb2gray_viaLab_NCHW_th(img_rgb/255.)*255).to(torch.uint8)

def rgb2gray_viaLab_NCHW_float_th(img_rgb, keepC3=True):
    if img_rgb.max() > 1.0001: raise ValueError(f"maximum value of image is > 1, should be 0..1")
    return _rgb2gray_viaLab_NCHW_th(img_rgb)


def _rgb2gray_viaLab_NCHW_th(img_rgb, keepC3=True):
    if img_rgb.shape[1] != 3 : raise ValueError(f"wrong dimensioniality should be [:,3,...] e.g. NCHW ,but is [{img_rgb.shape}]")
    
    img_lab = rgb2lab_NCHW(img_rgb)
    img_lab = img_lab.contiguous() # Seems to be necessary in pt1.8 otherwise a cuda memory error appears!
    z = torch.zeros_like(img_lab[:,0:2] )  # non normalized Lab space => zero point @ 0
    img_lab_nocol = torch.cat([img_lab[:,0:1], z], dim=1)
    img_rgb_nocol = lab2rgb_NCHW(img_lab_nocol)
    # max_delta = max((img_rgb_nocol[:,0:1] -img_rgb_nocol[:,1:2]).abs().max(), (img_rgb_nocol[:,0:1] -img_rgb_nocol[:,2:3]).abs().max())
    # print(max_delta)
    img_rgb_nocol = img_rgb_nocol.mean(dim=1,keepdim=True)
    img_rgb_nocol = torch.clamp(img_rgb_nocol, 0, 1)
    if keepC3:
        img_rgb_nocol = img_rgb_nocol.repeat(1,3,1,1)
    return img_rgb_nocol

def rgb2gray_viaLab_NHWC_uint8_np(img_rgb, keepC3=True):
    if img_rgb.dtype != np.uint8 : raise ValueError(f"wrong datatype")
    img_rgb_nocol = rgb2gray_viaLab_NHWC_np(img_rgb, keepC3)
    img_rgb_nocol = (img_rgb_nocol*255).astype(np.uint8)
    return img_rgb_nocol

def rgb2gray_viaLab_NHWC_np(img_rgb, keepC3=True):
    if img_rgb.shape[-1] != 3 : raise ValueError(f"wrong dimensioniality should be [...,3], e.g. NHWC,  but is [{img_rgb.shape}]")
    # if img_rgb.dtype != np.uint8 : raise ValueError(f"wrong datatype")
    img_lab = sk_rgb2lab(img_rgb)
    z = np.zeros_like(img_lab[...,0:1] )  # non normalized Lab space => zero point @ 0
    img_lab_nocol = np.concatenate([img_lab[...,0:1], z,z], axis=-1)
    img_rgb_nocol = sk_lab2rgb(img_lab_nocol)
    # max_delta = max(np.abs(img_rgb_nocol[...,0:1] -img_rgb_nocol[...,1:2]).max(), np.abs(img_rgb_nocol[...,0:1] -img_rgb_nocol[...,2:3]).max())
    # print(max_delta)  # max delta was ~ 5e-5 # => seems like numeric errors => take mean
    img_rgb_nocol = img_rgb_nocol.mean(axis=-1,keepdims=False)
    img_rgb_nocol = np.clip(img_rgb_nocol,0, 1)
    if keepC3:
        img_rgb_nocol = np.stack([img_rgb_nocol,img_rgb_nocol,img_rgb_nocol],axis=-1)
    return img_rgb_nocol

def rgb2lab_normalized_NCHW(srgb_NCHW):
    """converts from normalized rgb 2 lab [0,1]=>[0,1] 
     L_chan: grayscale  input range [0, 100]
     a_chan/b_chan: color channels with input range ~[-110, 110], not exact
     [0, 100] => [0, 1],  ~[-110, 110] => [0, 1]

     => the 0 point in the Lab space is on 0.5
    """
    if srgb_NCHW.shape[1] != 3 : raise ValueError(f"wrong dimensioniality should be [:,3,...] e.g. NCHW ,but is [{img_rgb.shape}]")
    lab_NHWC = rgb2lab_NHWC(srgb_NCHW.permute(0,2,3,1))
    lab_NCHW = lab_NHWC.permute(0,3,1,2)
    scales = torch.tensor([100,220,220],device=lab_NHWC.device, dtype=lab_NHWC.dtype)[None,:,None,None]
    ofs    = torch.tensor([  0,0.5,0.5],device=lab_NHWC.device, dtype=lab_NHWC.dtype)[None,:,None,None]
    lab_NCHW_norm = lab_NCHW/scales + ofs
    return lab_NCHW_norm

def lab_normalized_NCHW_to_std_lab_NCHW(lab_NCHW_norm):
    """ Converts normalized NCHW lab tensor to a un-normalized lab tensor
    """
    if lab_NCHW_norm.shape[1] != 3 : raise ValueError(f"wrong dimensioniality should be [:,3,...] e.g. NCHW ,but is [{img_rgb.shape}]")
    scales = torch.tensor([100,220,220],device=lab_NCHW_norm.device, dtype=lab_NCHW_norm.dtype)[None,:,None,None]
    ofs    = torch.tensor([  0,0.5,0.5],device=lab_NCHW_norm.device, dtype=lab_NCHW_norm.dtype)[None,:,None,None]
    lab_NCHW_std = (lab_NCHW_norm-ofs)*scales
    return lab_NCHW_std

def lab2rgb_normalized_NCHW(lab_norm_NCHW):    
    """converts from normalized lab 2 rgb [0,1]=>[0,1]
    
     L_chan: black and white with input range [0, 100]
     a_chan/b_chan: color channels with input range ~[-110, 110], not exact
     [0, 100] <= [0, 1],  ~[-110, 110] <= [0, 1]

     => the 0 point in the Lab space is on 0.5
     """
    scales = torch.tensor([100,220,220],device=lab_norm_NCHW.device, dtype=lab_norm_NCHW.dtype)[None,:,None,None]
    ofs    = torch.tensor([  0,0.5,0.5],device=lab_norm_NCHW.device, dtype=lab_norm_NCHW.dtype)[None,:,None,None]
    lab_NCHW = (lab_norm_NCHW - ofs) * scales
    srgb_NHWC = lab2rgb_NHWC(lab_NCHW.permute(0,2,3,1))
    return srgb_NHWC.permute(0,3,1,2)

def rgb2lab_NCHW(srgb_NCHW):
    lab_NHWC = rgb2lab_NHWC(srgb_NCHW.permute(0,2,3,1)) # [NCHW] > [NHWC]
    return lab_NHWC.permute(0,3,1,2)

def lab2rgb_NCHW(lab_NCHW):
    srgb_NHWC = lab2rgb_NHWC(lab_NCHW.permute(0,2,3,1))
    return srgb_NHWC.permute(0,3,1,2)


# Color conversion factors are from skimage:
# https://github.com/scikit-image/scikit-image/blob/515655f7029079bf95328bc5e83f9845cf6479eb/skimage/color/colorconv.py#L338
# and are basically identical (down to rounding errors) to 
# and also Book Szeliski 2010, Computer Vision: Algorithms and Applications page 83
_conv_mat_rgb2xyz = np.array([#       X        Y          Z             
                                [0.412453, 0.212671, 0.019334], # R   
                                [0.357580, 0.715160, 0.119193], # G
                                [0.180423, 0.072169, 0.950227], # B
                        ])
_conv_mat_xyz2rgb = np.linalg.inv(_conv_mat_rgb2xyz)
_conv_ilum_d65_obs10 = np.array([0.94809667673716, 1,  1.0730513595166162])
_conv_ilum_d65_obs10_inv = 1.0/_conv_ilum_d65_obs10
_conv_ilum_d65_obs2  =  np.array([0.95047, 1., 1.08883]) # SKimage Default d65_2
_conv_ilum_d65_obs2_inv = 1.0/_conv_ilum_d65_obs2
_conv_mat_nonlin_xyz2lab = np.array([   #  fx      fy      fz
                                    [  0.0,  116.0,    0.0], # L
                                    [500.0, -500.0,    0.0], # a
                                    [  0.0,  200.0, -200.0], # b
                                ])
_conv_ofs_nonlin_xyz2lab = np.array([-16.0, 0.0, 0.0])
_conv_mat_lab2xyz_nonlin = np.array([ #   fx      fy        fz
                                    [1/116.0, 1/116.0,  1/116.0], # l
                                    [1/500.0,     0.0,      0.0], # a
                                    [    0.0,     0.0, -1/200.0], # b
                                 ])


def rgb2xyz_NHWC(srgb):
    if srgb.max() > 1.0001: raise ValueError(f"maximum value of image is > 1 but should be [0...1]")
    if srgb.ndim !=4: raise ValueError(f"image shape must be [NHWC] but is {srgb.shape}")
    if srgb.shape[-1] !=3: raise ValueError(f"image shape must be [NHW3] but is {srgb.shape}")
    # compare with https://en.wikipedia.org/wiki/CIELAB_color_space
    dtype = srgb.dtype
    device = srgb.device
    srgb_pixels = torch.reshape(srgb, [-1, 3])

    # SRGB to CIE XYZ
    # https://en.wikipedia.org/wiki/SRGB
    # The RGB2XYZ conversion is identical to SKimage  https://github.com/scikit-image/scikit-image/blob/515655f7029079bf95328bc5e83f9845cf6479eb/skimage/color/colorconv.py#L657
    linear_mask = (srgb_pixels <= 0.04045).to(dtype=dtype)
    exponential_mask = (srgb_pixels > 0.04045).to(dtype=dtype)
    rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
    rgb_to_xyz = torch.from_numpy(_conv_mat_rgb2xyz).to(device=device,dtype=dtype)

    xyz_pixels = rgb_pixels @ rgb_to_xyz
    xyz_pixels = xyz_pixels.reshape(srgb.shape)
    return xyz_pixels

def xyz2lab_NHWC(xyz):
    if xyz.ndim !=4: raise ValueError(f"image shape must be [NHWC] but is {xyz.shape}")
    if xyz.shape[-1] !=3: raise ValueError(f"image shape must be [NHW3] but is {xyz.shape}")
    dtype = xyz.dtype
    device = xyz.device        
    xyz_pixels = xyz.reshape([-1, 3])

    # XYZ to Lab (Standard Illuminant D65:)
    std_ilum_d65_obs2_inv = torch.from_numpy(_conv_ilum_d65_obs2_inv).to(device=device, dtype=dtype)
    mat_fxfyfz_to_lab   = torch.from_numpy(_conv_mat_nonlin_xyz2lab ).to(device=device, dtype=dtype)
    ofs_fxfyfz_to_lab   = torch.from_numpy(_conv_ofs_nonlin_xyz2lab ).to(device=device, dtype=dtype)
    
    # convert from XYZ to normalized version xyz by normalizing with illuminant D65
    xyz_normalized_pixels = xyz_pixels * std_ilum_d65_obs2_inv[None,:]
    # perform non-linear scaling function
    # non linear function is a finite-slope approximation of the cube root  (Book Szeliski 2010, Computer Vision: Algorithms and Applications)
    #  0.008856   ~ (6/29)**3          |        { x**(1/3)                             if x  > (6/29)**3
    #  7.7870     ~ 1/(3 * (6/29)**2)  | f(x) = { 
    #  16.0 / 116 = 2/3*6/29           |        { x * 1/(3 *(6/29)**2) + 2/3 * (6/29)  if x <= (6/29)**3
    linear_mask   = (xyz_normalized_pixels <= (0.008856)).to(dtype=dtype)
    fxfyfz_pixels =      linear_mask * (  xyz_normalized_pixels * 7.7870 + 16.0/116.0              ) + \
                    (1-linear_mask) * (  torch.clamp_min(xyz_normalized_pixels, 1e-9) ** (1.0/3.0) )
                    #  (1-linear_mask) * (  xyz_normalized_pixels ** (1.0/3.0)             )
    lab_pixels = (fxfyfz_pixels @ (mat_fxfyfz_to_lab.T) ) + ofs_fxfyfz_to_lab[None,:]
    return torch.reshape(lab_pixels, xyz.shape)

def rgb2lab_NHWC(srgb):
    if srgb.shape[-1] !=3: raise ValueError(f"image shape must be [NHW3] but is {srgb.shape}")
    # compare with https://en.wikipedia.org/wiki/CIELAB_color_space
    # SRGB to CIE XYZ
    xyz = rgb2xyz_NHWC(srgb)
    # CIE XYZ to CIE Lab
    lab = xyz2lab_NHWC(xyz)
    return lab


def lab2xyz_NHWC(lab):
    if lab.ndim !=4: raise ValueError(f"image shape must be [NHWC] but is {lab.shape}")
    if lab.shape[-1] !=3: raise ValueError(f"image shape must be [NHW3] but is {lab.shape}")
    dtype = lab.dtype
    device = lab.device        
    lab_pixels = lab.reshape([-1, 3])

    std_ilum_d65_obs2 = torch.from_numpy(_conv_ilum_d65_obs2).to(device=device, dtype=dtype)
    conv_mat_lab2xyz_nonlin = torch.from_numpy(_conv_mat_lab2xyz_nonlin).to(device=device, dtype=dtype)
    conv_ofs_nonlin_xyz2lab = torch.from_numpy(_conv_ofs_nonlin_xyz2lab).to(device=device, dtype=dtype)

    fxfyfz_pixels = (lab_pixels - conv_ofs_nonlin_xyz2lab ) @ conv_mat_lab2xyz_nonlin
    fxfyfz_pixels = torch.clamp_min(fxfyfz_pixels, 0)

    # inverse of nonlinear function
    # => 0.2068966 ~ 6/29
    # perform inverse of non-linear scaling function
    # non linear function is a finite-slope approximation of the cube root  (Book Szeliski 2010, Computer Vision: Algorithms and Applications)
    #  0.008856   ~ (6/29)**3          |        { y**3                                 if y  > (6/29)
    #  7.787      ~ 1/(3 * (6/29)**2)  | g(y) = {  
    #  16.0 / 116 = 2/3*6/29           |        { (y- 2/3 *(6/29))  * (3 *(6/29)**2)   if y <= (6/29)
    linear_mask    = (fxfyfz_pixels <= (0.2068966)).to(dtype=dtype)
    xyz_normalized =       linear_mask  * ( (fxfyfz_pixels - 16.0/116.0) / 7.7870  ) + \
                        (1-linear_mask) * (  fxfyfz_pixels ** 3                    )

    # denormalize for D65 white point observer2
    xyz = xyz_normalized * std_ilum_d65_obs2
    # xyz = torch.clamp(xyz, 1e-9, 1)
    return xyz.reshape(lab.shape)


def xyz2rgb_NHWC(xyz):
    if xyz.ndim !=4: raise ValueError(f"image shape must be [NHWC] but is {xyz.shape}")
    if xyz.shape[-1] !=3: raise ValueError(f"image shape must be [NHW3] but is {xyz.shape}")
    dtype = xyz.dtype
    device = xyz.device        
    xyz_flat = xyz.reshape([-1, 3])

    # CIE XYZ to SRGB
    mat_xyz2rgb = torch.from_numpy(_conv_mat_xyz2rgb).to(device=device,dtype=dtype)

    rgb_flat = torch.clamp(xyz_flat @ mat_xyz2rgb, 1e-9, 1)

    mask = (rgb_flat > 0.0031308).to(dtype=dtype)
    srgb_flat = (  mask) * (1.055 * rgb_flat ** (1/2.4) - 0.055) + \
                (1-mask) * (         rgb_flat * 12.92          )
    return srgb_flat.reshape(xyz.shape)


def lab2rgb_NHWC(lab):
    if lab.shape[-1] !=3: raise ValueError(f"image shape must be [NHW3] but is {lab.shape}")
    # compare with https://en.wikipedia.org/wiki/CIELAB_color_space
    # CIE Lab to CIE XYZ
    xyz = lab2xyz_NHWC(lab)
    # CIE XYZ to standard RGB
    srgb = xyz2rgb_NHWC(xyz)
    return srgb
