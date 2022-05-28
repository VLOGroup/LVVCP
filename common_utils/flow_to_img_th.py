import numpy as np
import torch

UNKNOWN_FLOW_THRESH = 1e7
SMALLFLOW = 0.0
LARGEFLOW = 1e8
# from https://github.com/gengshan-y/VCN
def flow_to_image(flow, maxrad=None):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """

    def make_color_wheel():
        """
        Generate color wheel according Middlebury color code
        :return: Color wheel
        """
        RY = 15
        YG = 6
        GC = 4
        CB = 11
        BM = 13
        MR = 6

        ncols = RY + YG + GC + CB + BM + MR

        colorwheel = np.zeros([ncols, 3])

        col = 0

        # RY
        colorwheel[0:RY, 0] = 255
        colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
        col += RY

        # YG
        colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
        colorwheel[col:col+YG, 1] = 255
        col += YG

        # GC
        colorwheel[col:col+GC, 1] = 255
        colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
        col += GC

        # CB
        colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
        colorwheel[col:col+CB, 2] = 255
        col += CB

        # BM
        colorwheel[col:col+BM, 2] = 255
        colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
        col += + BM

        # MR
        colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
        colorwheel[col:col+MR, 0] = 255

        return colorwheel

    def compute_color(u, v):
        """
        compute optical flow color map
        :param u: optical flow horizontal map
        :param v: optical flow vertical map
        :return: optical flow in color code
        """
        [h, w] = u.shape
        img = np.zeros([h, w, 3])
        nanIdx = np.isnan(u) | np.isnan(v)
        u[nanIdx] = 0
        v[nanIdx] = 0

        colorwheel = make_color_wheel()
        ncols = np.size(colorwheel, 0)

        rad = np.sqrt(u**2+v**2)

        a = np.arctan2(-v, -u) / np.pi

        fk = (a+1) / 2 * (ncols - 1) + 1

        k0 = np.floor(fk).astype(int)

        k1 = k0 + 1
        k1[k1 == ncols+1] = 1
        f = fk - k0

        for i in range(0, np.size(colorwheel,1)):
            tmp = colorwheel[:, i]
            col0 = tmp[k0-1] / 255
            col1 = tmp[k1-1] / 255
            col = (1-f) * col0 + f * col1

            idx = rad <= 1
            col[idx] = 1-rad[idx]*(1-col[idx])
            notidx = np.logical_not(idx)

            col[notidx] *= 0.75
            img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

        return img
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    rad = np.sqrt(u ** 2 + v ** 2)
    if maxrad is None:
        maxrad = max(-1, np.max(rad))
       

    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)

#=======================================================================


def makeColorWheel_np():
    """
    This is a helper function that calculates the colorwheel that is used for
    optical flow visualization
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    size = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((3, size))

    col = 0
    # RY
    colorwheel[0, col:col+RY] = 255
    colorwheel[1, col:col+RY] = np.floor(255 * np.arange(RY)/RY)
    col += RY

    # YG
    colorwheel[0, col:col+YG] = 255 - np.floor(255 * np.arange(YG)/YG)
    colorwheel[1, col:col+YG] = 255
    col += YG

    # GC
    colorwheel[1, col:col+GC] = 255
    colorwheel[2, col:col+GC] = np.floor(255 * np.arange(GC)/GC)
    col += GC

    # CB
    colorwheel[1, col:col+CB] = 255 - np.floor(255 * np.arange(CB)/CB)
    colorwheel[2, col:col+CB] = 255
    col += CB

    # BM
    colorwheel[0, col:col+BM] = np.floor(255 * np.arange(BM)/BM)
    colorwheel[2, col:col+BM] = 255
    col += BM

    # MR
    colorwheel[0, col:col+MR] = 255
    colorwheel[2, col:col+MR] = 255 - np.floor(255 * np.arange(MR)/MR)

    return colorwheel.astype('uint8');



def computeNormalizedFlow( flow, valid_mask=None, max_flow=-1,  lower_auto_limit = 0,sqrt_eps=1e-9):
    """
    
    lower_auto_limit... if automatic normalization is used, this is  the lower limit  normalization (avoids to amplify noise)
    
    This function normalizes the optical flow field.
    
    Normalization means that the `max_flow` value is linerary scaled to be 1.
    `max_flow` can either be specified externally, or calculated from the maximum length of the flow vector.
    If a min_max_flow is specified (>0) then the maximum flow, then the estimated maximum flow will be no smaller than this value.
    This limits amplified visualization of small noise when no flow is present
    """   
    dtype, device = flow.dtype, flow.device
    if not (flow.ndim == 4 and flow.shape[1] == 2):
        raise ValueError(f"expected flow in N2HW format but is {flow.shape}")
    

    if valid_mask is not None:
        # valid_mask = torch.ones_like (flow[:,0:1])   
        if not (valid_mask.ndims == 4 and valid_mask.shape[1] in [1,2]):
            raise ValueError(f"expected valid mask in N1HW or N2HW format ")
        # calculate the length of the flow vectors (flow_mag) and select the longest
        f_masked_sq = (flow * valid_mask)**2
    else :
        f_masked_sq = flow **2
    
    flow_max_mag = torch.max( f_masked_sq[:, 0:1] +  f_masked_sq[:, 1:2] )
    flow_max_mag = torch.sqrt( torch.clamp_min(flow_max_mag , sqrt_eps))
    automax =  torch.clamp_min( flow_max_mag, lower_auto_limit )
    
    # if max_flow =-1 => use automax else the provided max_flow
    useMax_nAuto  = (max_flow >= 0)
    max_mag = torch.clamp_min( max_flow * useMax_nAuto + automax * (1.0-useMax_nAuto), sqrt_eps)
    
    normalized_flow = flow/max_mag
    return normalized_flow

def computeFlowImg_th(flow, valid_mask = None, max_flow=-1, lower_auto_limit = 0,sqrt_eps=1e-9):
    dtype,device = flow.dtype, flow.device
    normalized_flow = computeNormalizedFlow( flow, valid_mask, max_flow, lower_auto_limit, sqrt_eps)
    # f_masked_sq = normalized_flow **2
    # flow_mag = torch.sqrt( torch.clamp_min(   f_masked_sq[:,0:1] + f_masked_sq[:,1:2] , sqrt_eps))
    
    cw = makeColorWheel_np().astype(np.float32).T / 255.0
    cw = torch.from_numpy(cw).to(device=device,dtype=dtype)
    n_cols = cw.shape[0]

    normalized_flow = torch.clamp(normalized_flow,-1,1)
    u,v = normalized_flow[:,0:1], normalized_flow[:,1:2]
    mag = torch.norm(normalized_flow, dim=1, p=2, keepdim=True)
    phi = torch.atan2( -v, -u) / np.pi   # range [-1,1]
    phi = (phi+1.0)/2.0                  # range [0,1]
    phi_idx = phi * (n_cols -1)          # range [0, ncols-1]
    
    # bilinear weights
    f_phi_idx = torch.floor(phi_idx)
    c_phi_idx = f_phi_idx +1
    d_phi = phi_idx - f_phi_idx
    c_phi_idx  = torch.clamp( c_phi_idx, 0, n_cols -1)

    f_phi_idx = f_phi_idx.to(dtype=torch.long)
    c_phi_idx = c_phi_idx.to(dtype=torch.long)    
    
    # get colors via bilinear sampling from color wheel
    idc = torch.arange(3,device=device)[None,:,None,None]
    col_floor = cw[f_phi_idx, idc ]
    col_ceil  = cw[c_phi_idx, idc ]
    col =  (1.0 - d_phi)*col_floor + d_phi * col_ceil

    # Scale colors with magnitude (out of range is caled differently)
    mag_valid = (mag <= 1.0).to(mag.dtype)
    col = (1.0 - mag*(1.0-col) ) * mag_valid + (1-mag_valid) * col*0.75
    # torch.clamp(mag, 0, 1)
    # col  = 1 - mag * (1-col)
    
    col = torch.clamp(col, 0, 1)
    img = (col*255).to(dtype=torch.uint8)
    return img
        

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    H = 5
    W = 10
    idy = torch.arange(H)[None,None,:,None].repeat(1,1,1,W) - H/2 + 0.5
    idx = torch.arange(W)[None,None,None,:].repeat(1,1,H,1) - W/2 + 0.5

    flow = torch.cat([idx,idy],dim=1)
    img = computeFlowImg_th(flow)
    
    img = img.cpu().numpy()[0].transpose(1,2,0)
    plt.imshow(img)
    # plt.show()
    print ("done")

    from os.path import split, join
    import os, sys
    import imageio
    _lib =  os.path.dirname(os.path.abspath(__file__))+"/.."
    print(_lib)
    if _lib not in sys.path:
        print("adding {_lib} to system path")
        sys.path.insert(0, _lib)


    from common_utils.flow_io_imageio import load_Kitti_png_flow    


    path = split(__file__)[0]
    fn_flow_vis = "./data/00000__00001_04c_flowvis.png"
    fn_flow_vec = "./data/00000__00001_04d_flowvec.png"
    flow_vis =          imageio.imread(join(path, fn_flow_vis) )
    flow_vec = load_Kitti_png_flow(join(path, fn_flow_vec) )[0]
    flow_vec_th = torch.from_numpy(flow_vec[:,:,0:2]).to(dtype=torch.float32).permute(2,0,1)[None]

    flow_vis_old_np = flow_to_image(flow_vec)
    flow_vis_new_th = computeFlowImg_th(flow_vec_th, max_flow = -1)  #N2HW => N3HW
    # flow_vis_new_th = computeFlowImg_th(flow_vec_th, max_flow = 10)  #N2HW => N3HW
    flow_vis_new_np = flow_vis_new_th[0].cpu().numpy().transpose(1,2,0)

    fig,axes = plt.subplots(3,2, num=1, clear=True, sharex=True, sharey=True)
    plt.sca(axes[0,0])
    plt.imshow(flow_vis)
    plt.title("FlowVis from file")
    plt.sca(axes[0,1])
    plt.imshow( np.linalg.norm(flow_vec,axis=-1))
    plt.title("Flow Mag")
    plt.sca(axes[1,0])
    plt.imshow(flow_vis_new_np)
    plt.title("Flow Vis pytorch")
    plt.sca(axes[2,0])
    plt.imshow(flow_vis_old_np)
    plt.title("Flow Vis python")
    plt.sca(axes[1,1])
    plt.imshow( np.abs(flow_vis_new_np*1.0 - 1.0*flow_vis).mean(axis=-1) )
    plt.title("Delta File vs new TH")
    plt.sca(axes[2,1])
    plt.imshow( np.abs(flow_vis_old_np*1.0 - 1.0*flow_vis_new_np).mean(axis=-1) )
    plt.title("Delta Python vs new TH")

    plt.show()
    print ("done")


    





