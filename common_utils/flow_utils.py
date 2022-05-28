import numpy as np
import imageio
import torch
import torch.nn.functional as F


def bilinear_sampler(img, coords, mode:str='bilinear'):
    """ Wrapper for grid_sample, uses pixel coordinates 
        Bilinear Sampler wrapper from RAFT 
    """
    assert coords.shape[-1] == 2
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)

    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    return img

def bilinear_sampler_with_mask(img, coords, mode:str='bilinear'):
    """ Wrapper for grid_sample, uses pixel coordinates
        Bilinear Sampler wrapper from RAFT
    """
    assert coords.shape[-1] == 2
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    mask_valid = (xgrid > 0) & (ygrid > 0) & (xgrid < (W-1)) & (ygrid < (H-1))

    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    return img, mask_valid.float()

def coords_grid(batch: int, ht: int, wd: int):
    """ Simple Meshgrid wrappter from RAFT 
    """
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack((coords[1], coords[0]), dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)

def gen_confidence_np(i1w, i2, fdelta, is_Lab=False, eps_df=2, eps_dlum=0.4):
    """ Computes the confidence based 
    expect HWC memory layout """
    if len(fdelta.shape) == 2:
        fdelta = fdelta[...,None]

    assert i1w.shape[-1] in [1,2,3], f"numpy array expected to have colour in last axis"
    if is_Lab: #LAB
        d_lum = np.abs( i1w[...,0:1] - i2[...,0:1])
    else: # RGB
        d_lum = np.abs( (i1w - i2).mean(-1,keepdims=True) )

    c_lum =  np.maximum( eps_dlum - d_lum,0) / eps_dlum
    c_flow = np.maximum( eps_df - fdelta,0) / eps_df
    return c_lum * c_flow

def gen_confidence_th(i1w, i2, fdelta, is_Lab=False, eps_df=2, eps_dlum=0.4):
    """expect NCHW memory layout """
    assert type(i1w) == type(i2) == type(fdelta) == torch.Tensor 
    if len(fdelta.shape) == 2:
        fdelta = fdelta[:,None]
    assert len(i1w.shape) == len(i2.shape) == len(fdelta.shape) == 4
    assert i1w.shape[1] in [1,2,3], f"tensor expected to have colour in last axis"

    if is_Lab: #LAB
        d_lum = ( i1w[:,0:1] - i2[:,0:1]).abs()
    else: # RGB
        d_lum = ( (i1w - i2).abs().mean(dim=1,keepdims=True) )

    c_lum  = torch.clamp_min( eps_dlum - d_lum, 0) / eps_dlum
    c_flow = torch.clamp_min( eps_df   - fdelta,0) / eps_df
    return c_lum * c_flow


def flow2uint16restoresim(flow):
    """ Simulates storing and restoring of a flow tensor"""    
    if type(flow) == np.ndarray:
        flow_uint16 =     np.clip(flow*64 + 2**15, 0, (2**16)-1).astype(np.uint16)
        flow = (flow_uint16.astype(np.int32) - 2**15).astype(np.float64)/64.0
    elif type(flow) == torch.Tensor:
        flow_uint16 = torch.clamp(flow*64 + 2**15, 0, (2**16)-1).to(dtype=torch.int32)
        flow = (flow_uint16 - 2**15).to(dtype=flow.dtype)/64.0
    else:
        assert False, f"Unkwon datatype: {type(flow)}"
    return flow

def flow2uint16(flow):
    """Converts a flow vector from float to uint16 (similar to Kitti png) """
    if type(flow) == np.ndarray:
        flow_uint16 =     np.clip(flow*64 + 2**15, 0, (2**16)-1).astype(np.uint16)
    elif type(flow) == torch.Tensor:
        flow_uint16 = torch.clamp(flow*64 + 2**15, 0, (2**16)-1).cpu().numpy().astype(np.uint16)
    else:
        assert False, f"Unkwon datatype: {type(flow)}"
    return flow_uint16

def uint16flow2flow(uint16flow):
    """Converts a uint16 encoded flow vector back to float (similar to Kitti png) """
    assert uint16flow.dtype == np.uint16, f"flow needs to be of uint16 datatype!"
    return (uint16flow.astype(np.int32) - 2**15).astype(np.float64)/64.0


def l2sq(vec,dim: int=1):
    return torch.sum(vec**2,dim=dim)
    
def est_occl_heu(flow_fwd:torch.Tensor, flow_bwd:torch.Tensor, alpha1:float=0.01, alpha2:float=0.05, scale:float=1):
    flow_fwd = flow_fwd * scale
    flow_bwd = flow_bwd * scale
    flow_warped = flow_warp(flow_bwd, flow_fwd)
    delta = flow_fwd + flow_warped
    flow_mag_sq = l2sq(flow_fwd) + l2sq(flow_warped)
    occl = l2sq(delta) > (alpha1 * flow_mag_sq + alpha2)
    return delta, occl


def flow_warp_pt_wrapped_np(x, flo):
    assert len(x.shape) ==3
    assert len(flo.shape) ==3
    assert flo.shape[-1]== 2
    assert flo.shape[0:2] == x.shape[0:2]
    x_th = torch.from_numpy(  x.transpose(2,0,1))[None,...].float()
    f_th = torch.from_numpy(flo.transpose(2,0,1))[None,...].float()
    xw_th = flow_warp(x_th, f_th)
    xw = xw_th.cpu().numpy().transpose(1,2,0)
    return xw

def flow_warp_with_mask(x:torch.Tensor, flo:torch.Tensor, align_corners:bool=True):
    """
    inverse warp an image/tensor (im2) back to im1, according to the optical flow

    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow

    returns warped_output, valid_mask

    """
    if not torch.jit.is_scripting():
        assert type(x) == type(flo) == torch.Tensor, f"only implemented for torch tensors"
    assert flo.shape[1] == 2, f"flow shape must be N2HW"
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1).to(x.device, dtype=x.dtype)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W).to(x.device, dtype=x.dtype)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1)

    vgrid = grid + flo

    # scale grid to [-1,1]
    vgrid = torch.stack([
        2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0,
        2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
    ],
                        dim=1)

    vgrid = vgrid.permute(0, 2, 3, 1)
    # use same align_corners mode as in bilinear_sample used for sampling from cost volume in RAFT
    output = F.grid_sample(x, vgrid, padding_mode='border', align_corners=align_corners)
    valid_mask = torch.ones_like(x)
    valid_mask = F.grid_sample(valid_mask, vgrid, padding_mode='zeros', align_corners=align_corners)

    valid_mask[valid_mask < 0.9999] = 0  # allow a distance due to numerical errors.
    valid_mask[valid_mask > 0] = 1

    return output, valid_mask

def flow_warp(x:torch.Tensor, flo:torch.Tensor, align_corners:bool=True):
    warp, valid_mask = flow_warp_with_mask(x, flo, align_corners=align_corners)
    return warp * valid_mask

