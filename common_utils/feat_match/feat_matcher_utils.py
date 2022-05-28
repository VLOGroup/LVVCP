from multiprocessing.sharedctypes import Value
import warnings
import numpy as np
import torch
from torch.nn import functional as F
from ..flow_utils import bilinear_sampler, bilinear_sampler_with_mask, coords_grid, flow_warp_with_mask
from common_utils.torch_script_logging import log_info

try:
    from sample_cv_op import sample_cv
except:
    print('Did not find custom sampling operator sample_cv - skipping')

import matplotlib.pyplot as plt

from typing import Optional, List, Tuple


# As TorchScript doesn't correctly support the built in reverse
# function we needed to add or own version.
def torchscript_reversed(x:List[torch.Tensor]):
    reversed_x = []
    for i in range(len(x) - 1, -1, -1):
        reversed_x += [x[i]]

    return reversed_x


def build_sample_cv(flow, f1, f2, rx:int=2, ry:int=2, stride:Optional[int]=None, align_corners:bool=True):
    """ A simplified pytorch version of a differential cost-volume
        Replicates f2 features over a grid and uses batch matmul to compute cosine distance
    """
    N,C,H1,W1 =  f1.shape
    c_coords0 = coords_grid(N, H1, W1).to(device=f1.device)
    c_coords1 = c_coords0 + flow    

    f2_all_cands = sample_with_grid_HWdydx(f2,  c_coords1,ry=ry,rx=rx, stride=stride)  # N C H W ry rx
    f2_all_cands = f2_all_cands.permute(0,2,3,4,5,1).reshape(N*H1*W1,(2*ry+1)*(2*rx+1),C) #[N*H*W, dy*dx ,C]
    f1_comp = f1.permute(0,2,3,1).reshape(N*H1*W1,C,1)                                    #[N*H*W,   C   ,1]
    cv = torch.bmm(f2_all_cands,f1_comp).reshape(N,H1, W1, (2*ry+1), (2*rx+1))           # [N,H,W,dy,dx]
    return cv # [N,H,W,dy,dx]

def get_argmax_NHW_xy_last_np(arr):
    """ gets value and position of maximum element over the last two dimensions
       max_val,(max_idx,max_idy) = get_argmax_NHW_xy_last_th(tensor) """
    H,W = arr.shape[-2:]
    arr = arr.reshape(arr.shape[0:-2] + (H*W,))
    value  = np.max(arr,axis=-1)
    id_max = np.argmax(arr,axis=-1)
    id_y = id_max //W
    id_x = id_max %W
    return value, (id_x, id_y)

def get_argmax_NHW_xy_last_th(tens, stack_dim:int=0):
    """ gets value and position of maximum element over the last two dimensions
       max_val,(max_idx,max_idy) = get_argmax_NHW_xy_last_th(tensor) """
    H,W = tens.shape[-2:]
    tens = tens.reshape(tens.shape[0:-2] + (H*W,))
    value, id_max  = torch.max(tens,dim=-1)
    id_y = id_max //W
    id_x = id_max %W
    return value, torch.stack( (id_x, id_y), dim=stack_dim)

def gen_2D_mg_xy_th(H: int, W: int,  coord_axis: int=0, device: Optional[torch.device]=None):
    """ Generates a grid [1,2,H,W]. 
        The coordinates are stored:
            x in [:,0,...] 
            y in [:,1,...]
    """
    dx = torch.arange(0, W, device=device)
    dy = torch.arange(0, H, device=device)
    mgy, mgx = torch.meshgrid([dy,dx])  #python is xy pytorch is ij=yx
    coords_xy = torch.stack([mgx,mgy],dim=coord_axis)[None,...]
    return coords_xy

class CostVolume(torch.nn.Module):
    @staticmethod
    def fupnearest(x, ds: float):
        return torch.nn.functional.interpolate(x, scale_factor=ds, mode="nearest") * ds

    @staticmethod
    def fupbilin(x, ds: float):
        return torch.nn.functional.interpolate(x, scale_factor=ds, mode="bilinear", align_corners=True) * ds

    @staticmethod
    def fupbilin_noALC(x, ds: float):
        return torch.nn.functional.interpolate(x, scale_factor=ds, mode="bilinear", align_corners=False) * ds

    @staticmethod
    def fupbicubic(x, ds: float):
        return torch.nn.functional.interpolate(x, scale_factor=ds, mode="bicubic") * ds

    @staticmethod
    def cupnearest(x, ds: float):
        return torch.nn.functional.interpolate(x, scale_factor=ds, mode="nearest")

    @staticmethod
    def cupbilin(x, ds: float):
        return torch.nn.functional.interpolate(x, scale_factor=ds, mode="bilinear", align_corners=True)

    @staticmethod
    def cupbilin_noALC(x, ds: float):
        return torch.nn.functional.interpolate(x, scale_factor=ds, mode="bilinear", align_corners=False)

    @staticmethod
    def cupbicubic(x, ds: float):
        return torch.nn.functional.interpolate(x, scale_factor=ds, mode="bicubic")


    def __init__(self, use_autocast:bool=False, fwdbwd:bool=False, sample_cv_op_type:str='PyTorch', feat_upsample:str='nearest', rx:int=2, ry:int=2,  multi:bool=True, cmb_confs:bool=True):
        """ 
        Compute Full Cost-Volume using matmul, and returns winner takes all (WTA) matched positions and confidences

        if fwdbwd is True, the search is repeated for fmap2 => fmap1 at the same time without the need to recompute the cost-volume
        """
        super(CostVolume, self).__init__()

        self.use_autocast:bool = use_autocast
        self.fwdbwd:bool = fwdbwd
        self._sample_cv_op_type:str = sample_cv_op_type
        self._sample_cv_op_PyTorch:bool = False
        self._sample_cv_op_samplecv_NCHW:bool = False
        self._sample_cv_op_samplecv_NHWC:bool = False
        self._feat_upsample:str=feat_upsample
        self.rx:int = rx
        self.ry:int = ry

        self.cmb_confs = cmb_confs
        self.multi = multi

        if feat_upsample == "nearest":
            self.fup = self.fupnearest
            self.cup = self.cupnearest
        elif feat_upsample == "bilinear":
            self.fup = self.fupbilin
            self.cup = self.cupbilin
        elif feat_upsample == "bilinear_noALC":
            self.fup = self.fupbilin_noALC
            self.cup = self.cupbilin_noALC
        elif feat_upsample == "bicubic":
            self.fup = self.fupbicubic
            self.cup = self.cupbicubic
        else :
            raise ValueError(f"config['upsample'] must be 'nearest', 'bilinear' or  but is: '{feat_upsample}'")

        self._sample_cv_op_bilinear=True
        log_info(f"Matcher using: {self._sample_cv_op_type}")
        if self._sample_cv_op_type == 'PyTorch':
            self._sample_cv_op_PyTorch=True 
        else:
            if self._sample_cv_op_type.endswith('_nearest'):
                self._sample_cv_op_bilinear = False
                self._sample_cv_op_type= self._sample_cv_op_type[0:-len('_nearest')]

            if self._sample_cv_op_type == 'samplecv_NCHW':
                self._sample_cv_op_samplecv_NCHW=True
            elif self._sample_cv_op_type == 'samplecv_NHWC':
                self._sample_cv_op_samplecv_NHWC=True
            else:
                raise ValueError(f"Unkown setting for sample_cv_op_type={self._sample_cv_op_type}")

        self.legacy_mode:bool=False # Freezes Bugs for backward compatibility

    def compute_full_cv(self, fmap1:torch.Tensor, fmap2:torch.Tensor):
        if self.use_autocast:
            cv_full_h1w1h2w2 = self._compute_full_cv_w_autocast(fmap1, fmap2)
        else:
            cv_full_h1w1h2w2 = self._compute_full_cv(fmap1, fmap2)
        return cv_full_h1w1h2w2

    def _compute_full_cv(self, fmap1:torch.Tensor, fmap2:torch.Tensor):
        """
        Computes a full Cost-Volume by using bmm (batch matrix multiply)
        """
        B,C1,H1,W1 = fmap1.shape
        B,C2,H2,W2 = fmap2.shape
        assert C1 == C2 and H1== H2 and W1==W2
        fmap1 = fmap1.view(B, C1, H1*W1)
        fmap2 = fmap2.view(B, C2, H2*W2)   
        cv_full = torch.matmul(fmap1.transpose(1,2), fmap2)  # [B,H1*W1,H2*W2]
        cv_full_h1w1h2w2 = cv_full.reshape(B, H1,W1,H2,W2).clone()
        return cv_full_h1w1h2w2

    @torch.jit.unused
    def _compute_full_cv_w_autocast(self, fmap1:torch.Tensor, fmap2:torch.Tensor):
        """
        Wrapper for CV generation - does not work with TorchScript export
        This 
        """
        with (torch.cuda.amp.autocast(True)):
            cv_full_h1w1h2w2 = self._compute_full_cv(fmap1, fmap2)
        return cv_full_h1w1h2w2 


    def _mask_cost_volume(self, cv_full_h1w1h2w2:torch.Tensor, flow12: torch.Tensor, dx:int, dy:int):
        """ Masks out regions from the cost-volume, that are too far away
            Estimated center is given by the optical flow, and dx,dy specify maximum distances

            This function assumes that h1w1 is the image space and h2w2 the search space, where also
            the optical flow points to.
        """
        dtype = cv_full_h1w1h2w2.dtype
        device = cv_full_h1w1h2w2.device
        [N,H1,W1,H2,W2] = cv_full_h1w1h2w2.shape

        coords_xy = gen_2D_mg_xy_th(H2,W2,device=device).to(dtype=dtype) # [N,2,H2,W2]
    
        dxy =  torch.tensor([dx,dy],dtype=dtype,device=device)[None,:, None,None,None,None]
        ofs_xy_12 = coords_xy#  [0...W][0...H] =>  [-(W//2,...,0,...,W//2]-  torch.tensor([W2//2, H2//2],dtype=coords_xy.dtype,device=device)[None,:, None,None] # [N,2,H2,W2]
        ofs_xy_12 = ofs_xy_12[:,:,None,None,:,:] - coords_xy[:,:,:,:,None,None]  # Make a delta grid in the search space centered at most 0ofs point
        if flow12 is not None:
            ofs_xy_12 = ofs_xy_12 - flow12[:,:,:,:,None,None]  # [N,2, H2,W2, sy1,sx1] shift delta grid in search space by flow
        mask12 = (ofs_xy_12.abs() < dxy)
        mask12 = mask12[:,0] & mask12 [:,1]
        cv_full_h1w1h2w2 = cv_full_h1w1h2w2*mask12 #  # [N,H1,W1,H2,W2]
        return cv_full_h1w1h2w2


    def get_wta_conf(self, fmap1, fmap2, local_wdw:int=0, flow12_low:Optional[torch.Tensor]=None, flow21_low:Optional[torch.Tensor]=None, cand_cnt:int=8):
        """Computes the WTA (Winner Takes All) Solution over the whole cost-volume
           or alternatively over a restriced local neighborhood of the cost volume
        """
        B,C1,H1,W1  = fmap1.shape
        B,C2,H2,W2  = fmap2.shape
        device = fmap1.device
        dtype  = fmap1.dtype
        assert H1==H2 and W1 == W2, f"currently only done for identically sized images"

        cv_full_h1w1h2w2 = self.compute_full_cv(fmap1, fmap2) # [N,H1,W1,1,H2,W2]
        cv_full_h1w1h2w2.detach()  # [N,H1,W1,1,H2,W2]

        # Normalize CV (linear):
        cv_full_h1w1h2w2 = torch.clamp_min( cv_full_h1w1h2w2, 0.001)

        cv_full_h2w2h1w1 = torch.empty(size=(0,),dtype=dtype,device=device)
        if self.fwdbwd:
            cv_full_h2w2h1w1 = cv_full_h1w1h2w2.permute(0,3,4,1,2).detach().contiguous()  # [N,H1,W1,1,H2,W2] =>  [N,H2,W2,1,H1,W1]

        coords_xy = gen_2D_mg_xy_th(H2,W2,device=device).to(dtype=dtype) # [N,2,H2,W2]

        if local_wdw > 0:
            # Blank out not relevant parts of the CostVolume
            dx = dy = local_wdw
            cv_full_h1w1h2w2 = self._mask_cost_volume(cv_full_h1w1h2w2, flow12_low, dx, dy)
            if self.fwdbwd:
                cv_full_h2w2h1w1 = self._mask_cost_volume(cv_full_h2w2h1w1, flow21_low, dx, dy)

        conf12_wta, flow12_wta_idx = torch.topk(cv_full_h1w1h2w2.reshape([B,H1,W1,H2*W2]), cand_cnt, dim=-1, largest=True)
        flow12_wta = torch.stack([flow12_wta_idx % W2,flow12_wta_idx //W2],dim=1) - coords_xy[...,None]# [N,2,H1,W1,K]
        flows_12_wta = [flow12_wta[...,i]       .clone() for i in range(flow12_wta.shape[-1]) ]
        confs_12_wta = [conf12_wta[:,None,...,i].clone() for i in range(conf12_wta.shape[-1]) ]

        if self.fwdbwd:
            conf21_wta, flow21_wta_idx = torch.topk(cv_full_h2w2h1w1.reshape([B,H2,W2,H1*W1]), cand_cnt, dim=-1, largest=True)
            flow21_wta = torch.stack([flow21_wta_idx % W1,flow21_wta_idx //W1],dim=1) - coords_xy[...,None] # [N,2,H1,W1,K]
            flows_21_wta = [flow21_wta[...,i]       .clone() for i in range(flow21_wta.shape[-1]) ]
            confs_21_wta = [conf21_wta[:,None,...,i].clone() for i in range(conf21_wta.shape[-1]) ]
        else:
            flows_21_wta = [torch.empty(size=(),device=device,dtype=dtype)] # empty placeholder
            confs_21_wta = [torch.empty(size=(),device=device,dtype=dtype)] # empty placeholder

        return (flows_12_wta, flows_21_wta), (confs_12_wta, confs_21_wta), (None,None)


    # def refine_cands(self, input_gray, reference_gray, reference_col, motion_flow_fwd: Optional[torch.Tensor], motion_flow_bwd: Optional[torch.Tensor]):
    def refine_cands(self, flows_1to2_wta:List[torch.Tensor], confs_1_wta:List[torch.Tensor], input_feats:List[torch.Tensor], reference_feats:List[torch.Tensor]):
        """
        This method receives a coarse flow + confidence, and a pyramid of input and reference (to be matched) features and returnes a refined flow
        """
        _,_,H1,W1 = confs_1_wta[0].shape
        _,_,H1ds,W1ds = flows_1to2_wta[0].shape
        assert H1/H1ds == W1/W1ds, f"Img {[H1,W1]} vs {[H1ds,W1ds]} uses different ds factors {H1/H1ds} != {W1/W1ds}"
        ds = H1/H1ds
        device = confs_1_wta[0].device
        dtype = confs_1_wta[0].dtype

        feats1 = [f for f in input_feats]
        feats2 = [f for f in reference_feats]
        f1_old, f2_old = feats1[-1], feats2[-1]
        N = feats1[0].shape[0]

        if self.multi: # do a multistep refinement
            l_fsearch = len(feats1)-1 # drop lowest resolution, already matched
            feats1_search = torchscript_reversed(feats1[:-1]) # Torchscript does not support ::-1 so use pythons reversed iterator 
            feats2_search = torchscript_reversed(feats2[:-1]) # Torchscript does not support ::-1 so use pythons reversed iterator
        else:
            # go directly from lowest to highest resolution
            l_fsearch = 1
            feats1_search = feats1[0:1]
            feats2_search = feats2[0:1]

        flowcand = []
        confref  = []
        for i_search, (f1,f2) in enumerate(zip(feats1_search,feats2_search)):
            is_lastrun = (i_search == l_fsearch-1)
            
            _,_,c_H1ds,c_W1ds = f1_old.shape
            _,c_C,c_H1,c_W1 = f1.shape
            assert c_H1/c_H1ds == c_W1/c_W1ds == int(c_W1/c_W1ds)
            c_ds = c_H1/c_H1ds
            
            for i_flow, (c_flow, c_conf) in enumerate(zip(flows_1to2_wta, confs_1_wta)):
                c_flow = self.fup(c_flow,c_ds)
                c_conf = self.cup(c_conf,c_ds)

                if self._sample_cv_op_PyTorch:
                    c_search_conf = build_sample_cv(c_flow, f1, f2, rx=self.rx, ry=self.ry, stride=None, align_corners=True)
                elif self._sample_cv_op_samplecv_NCHW:
                    c_search_conf = sample_cv(f1, f2, c_flow, rx=self.rx, ry=self.ry, NCHW_nNHWC=True, bilinear=self._sample_cv_op_bilinear)
                elif self._sample_cv_op_samplecv_NHWC:
                    c_search_conf = sample_cv(f1, f2, c_flow, rx=self.rx, ry=self.ry, NCHW_nNHWC=False, bilinear=self._sample_cv_op_bilinear)
                else:
                    raise ValueError(f"Configuration Error for sampling operation")

                c_search_conf = torch.clamp_min(c_search_conf, 0.001)
                conf_best, delta_coords_best = get_argmax_NHW_xy_last_th(c_search_conf, stack_dim=1)
                delta_flow_best = delta_coords_best - torch.tensor([self.rx, self.ry], device=device, dtype=dtype)[None,:,None,None] 

                flow_best = c_flow + delta_flow_best
                if self.cmb_confs:
                    conf_best = conf_best * c_conf

                flows_1to2_wta[i_flow] = flow_best
                confs_1_wta[i_flow]    = conf_best

                if is_lastrun:
                    flowcand += [flow_best]
                    confref  += [conf_best]
                    
            f1_old, f2_old = f1, f2
        return flowcand, confref

    def sample_colcands(self, flowcand:List[torch.Tensor], confref:List[torch.Tensor], reference_col:torch.Tensor):
        """ Uses the extracted flow candidates + according confidences, and samples from the color reference image.
        """
        colcand = []
        for i, (c_flowcand, c_conf) in enumerate(zip(flowcand, confref)):
            ref_col_warped, valid_ooim  = flow_warp_with_mask(reference_col, c_flowcand, align_corners=True)
            # 
            if self.legacy_mode:
                # Frozen Bug for legacy compatibility mode
                #   clears invalid pixels, ignoring that the 0-point is actually at 0.5 not at 0
                ref_col_warped = (ref_col_warped) * valid_ooim 
            else:
                ref_col_warped = 0.5 + (ref_col_warped - 0.5) * valid_ooim  # 0 point in normalized Lab color space is at 0.5 
            confref[i] = c_conf * valid_ooim[:,0:1]     # if sampling is not 100% inside image => drop confidence, but keep color, let the regularizer decide
            colcand.append(ref_col_warped)
        return colcand, confref

    def __repr__(self):
        msg  = (f"CostVolume ("
                f"use_autocast={self.use_autocast},"
                f"fwdbwd={self.fwdbwd},"
                f"sample_cv_op_type='{self._sample_cv_op_type}',"
                f"feat_upsample='{self._feat_upsample}',"
                f"rx={self.rx},"
                f"ry={self.ry},"
                f"multi={self.multi},"
                f"cmb_confs={self.cmb_confs},"
                f")")
        return msg


def sample_with_grid_HWdydx(data, coords_ofs_xy, rx:int=8, ry:int=8, stride:Optional[int]=None):
    """
    Get a local Cost Volume on the finest resolution
    """
    assert coords_ofs_xy.shape[1] == 2, f"coords_ofs_xy must be in N2HW format"
    assert coords_ofs_xy.shape[0] == data.shape[0]
    N, _, H1, W1 = coords_ofs_xy.shape
    C,H2,W2 = data.shape[1:4]
    device = coords_ofs_xy.device
    dW2 = 2*rx+1
    dH2 = 2*ry+1
    
    dx = torch.linspace(-rx, rx, dW2, device=device)
    dy = torch.linspace(-ry, ry, dH2, device=device)
    if stride is not None:
        dx = dx*stride
        dy = dy*stride
    mg_y, mg_x = torch.meshgrid(dy, dx)

    centroid_HW = coords_ofs_xy.permute(0, 2, 3, 1)  # [N,2,H1,W1] => [N,H1,W1,2]
    centroid_HW = centroid_HW.reshape(N,1,1,H1,W1,2) # [N,1,1,H1,W1,2] # x,y
    delta_mg_dydx = torch.stack([mg_x, mg_y],dim=-1)
    delta_mg_dydx = delta_mg_dydx.reshape(1, dH2, dW2, 1, 1,2)
    coords_dydxHW = centroid_HW + delta_mg_dydx # [N,dH2,dW2,H1,W1,2]

    data_dup = data[:,None,None,].repeat(1,dW2,dH2,1,1,1)  # [N,dH2,dW2,C,H1,W1]

    coords_dydxHW = coords_dydxHW.reshape(N*dH2*dW2,  H1,W1,2) #   [N*dH2*dW2,H1,W1,2]
    data_dup      = data_dup     .reshape(N*dH2*dW2,C,H1,W1) #   [N*dH2*dW2,H1,W1,2]
    coords_dydxHW = coords_dydxHW.contiguous()
    data_dup      = data_dup.contiguous()

    data_grid_dydxHWd = bilinear_sampler(data_dup, coords_dydxHW)  #[N*dH2*dW2,C,H1,W1] => [N*dH2*dW2,C,H1,W1]
    data_grid_dydxHWd = data_grid_dydxHWd.reshape(N, dH2, dW2, C, H1, W1)  #[N*dH2*dW2,C,H1,W1] => [N, dH2, dW2, C, H1, W1]
    data_grid_HWdydx = data_grid_dydxHWd.permute(0,3,4,5,1,2) # [N, dH2, dW2, C, H1, W1] = > [N, C, H1, W1, dH2, dW2]
    return data_grid_HWdydx


def sample_with_grid_HWdydx_with_mask(data, coords_ofs_xy, rx:int=8, ry:int=8, stride:Optional[int]=None):
    """
    Get a local Cost Volume on the finest resolution
    """
    assert coords_ofs_xy.shape[1] == 2, f"coords_ofs_xy must be in N2HW format"
    assert coords_ofs_xy.shape[0] == data.shape[0]
    N, _, H1, W1 = coords_ofs_xy.shape
    C,H2,W2 = data.shape[1:4]
    device = coords_ofs_xy.device
    dW2 = 2*rx+1
    dH2 = 2*ry+1

    dx = torch.linspace(-rx, rx, dW2, device=device)
    dy = torch.linspace(-ry, ry, dH2, device=device)
    if stride is not None:
        dx = dx*stride
        dy = dy*stride
    mg_y, mg_x = torch.meshgrid(dy, dx)

    centroid_HW = coords_ofs_xy.permute(0, 2, 3, 1)  # [N,2,H1,W1] => [N,H1,W1,2]
    centroid_HW = centroid_HW.reshape(N,1,1,H1,W1,2) # [N,1,1,H1,W1,2] # x,y
    delta_mg_dydx = torch.stack([mg_x, mg_y],dim=-1)
    delta_mg_dydx = delta_mg_dydx.reshape(1, dH2, dW2, 1, 1,2)
    coords_dydxHW = centroid_HW + delta_mg_dydx # [N,dH2,dW2,H1,W1,2]

    data_dup = data[:,None,None,].repeat(1,dW2,dH2,1,1,1)  # [N,dH2,dW2,C,H1,W1]

    coords_dydxHW = coords_dydxHW.reshape(N*dH2*dW2,  H1,W1,2) #   [N*dH2*dW2,H1,W1,2]
    data_dup      = data_dup     .reshape(N*dH2*dW2,C,H1,W1) #   [N*dH2*dW2,H1,W1,2]
    coords_dydxHW = coords_dydxHW.contiguous()
    data_dup      = data_dup.contiguous()

    data_grid_dydxHWd, mask_valid_dydxHWd = bilinear_sampler_with_mask(data_dup, coords_dydxHW)  #[N*dH2*dW2,C,H1,W1] => [N*dH2*dW2,C,H1,W1]
    data_grid_dydxHWd = data_grid_dydxHWd.reshape(N, dH2, dW2, C, H1, W1)  #[N*dH2*dW2,C,H1,W1] => [N, dH2, dW2, C, H1, W1]
    mask_valid_dydxHWd = mask_valid_dydxHWd.reshape(N, dH2, dW2, 1, H1, W1)
    data_grid_HWdydx = data_grid_dydxHWd.permute(0,3,4,5,1,2) # [N, dH2, dW2, C, H1, W1] = > [N, C, H1, W1, dH2, dW2]
    mask_valid_HWdydx = mask_valid_dydxHWd.permute(0,3,4,5,1,2) # [N, dH2, dW2, C, H1, W1] = > [N, C, H1, W1, dH2, dW2]
    return data_grid_HWdydx, mask_valid_HWdydx