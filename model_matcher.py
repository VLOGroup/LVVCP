import torch

from argparse import Namespace

from common_utils.feat_match.feat_matcher import get_global_feat_matcher
from common_utils.flow_utils import coords_grid, est_occl_heu, flow_warp, flow_warp_with_mask
from common_utils.utils import get_min_cand_v1
from common_utils.torch_script_logging import log_warning

try:
    from sample_cv_op import sample_cv
except:
    print('Did not find custom sampling operator sample_cv - skipping')

from RAFT_custom.core.raft import RAFT

from typing import Optional, List

# As TorchScript doesn't correctly support the built in reverse
# function we needed to add or own version.
def reversed(x:List[torch.Tensor]):
    reversed_x = []
    for i in range(len(x) - 1, -1, -1):
        reversed_x += [x[i]]

    return reversed_x


class FeatureMatcher(torch.nn.Module):
    def __init__(self, config, logger):
        super(FeatureMatcher, self).__init__()
        self.feat_matcher = get_global_feat_matcher(backbone=config['backbone'], fwdbwd=False, config=config)

    def forward(self, input_gray, references_gray: List[torch.Tensor], references_col: List[torch.Tensor], motion_flows_fwd: Optional[List[torch.Tensor]]=None, motion_flows_bwd: Optional[List[torch.Tensor]]=None, references_masks: Optional[List[torch.Tensor]]=None):
        """ Predicts a set of color candidates for a gray input image, based on grayscale feature similarity to a set of reference images.

            input_gray: the main target to be colorized
            references_gray: a list of reference image tensors in gray-scale (used for feature matching)
            references_col:  a list of reference image tensors in color (used to extract the colors)
            ---
            if the module is used with 'useFlowOfs':True setting, then the local search window is offset by motion fields to be provided
            motion_flows_fwd: a list of optical flows from 
            motion_flows_bwd: a list of optical flows from 
        """
        input_gray = input_gray * 255

        candidates = []
        confidences = []
        motion_flow_fwd: Optional[torch.Tensor] = None
        motion_flow_bwd: Optional[torch.Tensor] = None
        ref_mask: Optional[torch.Tensor] = None
        for i, (ref_gray, ref_col) in enumerate(zip(references_gray, references_col)):
            if motion_flows_fwd is not None:
                motion_flow_fwd = motion_flows_fwd[i]
            if motion_flows_bwd is not None:
                motion_flow_bwd = motion_flows_bwd[i]
            if references_masks is not None:
                ref_mask = references_masks[i]

            ref_gray = ref_gray * 255

            candidates_tmp, confidences_tmp, flowcand_tmp   = self.feat_matcher(input_gray, ref_gray, ref_col, motion_flow_fwd, motion_flow_bwd)

            if ref_mask is not None:              
                log_warning("EXPERIMENTAL FEATURE - Alpha masking of reference")  
                for j,(f,c) in enumerate( zip(flowcand_tmp, confidences_tmp)):
                    ref_mask_warped  = flow_warp(ref_mask, f)
                    confidences_tmp[j] = (ref_mask_warped * c)
          
            candidates += candidates_tmp
            confidences += confidences_tmp

        # Stack them for easier compute
        candidates  = torch.stack(candidates, dim=-1)
        confidences = torch.stack(confidences, dim=-1)

        # Normalize CostVolume correlation confidence
        confidences = torch.clamp_min(confidences, 1e-6)  # remove negative correlation results
        confidences = torch.clamp(1 - confidences, 0 ,1)       # best = min

        best_candidate, best_confidence = get_min_cand_v1(candidates, confidences)

        candidates = candidates.permute(0,4,1,2,3) # Cands     [N,C,H,W,K] => [N,K,C,H,W]

        return candidates, best_candidate, best_confidence

class RaftMatcher(torch.nn.Module):
    def __init__(self, config, logger):
        super(RaftMatcher, self).__init__()
        raft_args = Namespace(**config)
        self.raft = RAFT(raft_args)
        if not self.raft.fwdbwd:
            raise ValueError(f"RaftMatcher requires that fwdbwd option for RAFT is activated in the config! {config}")
        # md5sum   models/raft-sintel.pth     cc69e5da1f38673ab10d1849859ebe91
        raft_ckpt = {k.replace('module.',''):v for k,v in torch.load(raft_args.model, map_location='cpu').items() if k.startswith('module') }
        self.raft.load_state_dict(raft_ckpt)
        self.raft.cuda()
        self.raft.eval()

        self.iters = config['iters']
        self.occl_a1 = config['occl_a1']
        self.occl_a2 = config['occl_a2']

        logger.info(f"Restoring RAFT Optial flow from {raft_args.model}, using {raft_args.iters} iters") 

    def forward(self, input_gray, ref_gray, ref_col):
        """
        Computes a color proposal for the input_gray based on RAFT optical flow which estimates motion.

        Returns 
            ref_col_warped: a tensor containing the warped color candidate, with colors blanked out for out of image regions
            flow: the estimated flow at high resolution
            flow_low: the estimated flow at 1/8 resolution
            flowdelta: the difference of the fwd/bwd optical flows after fwd/bwd check
            confidence: a heuristically estimated mask predicting which regions are matched
        """
        if not self.raft.fwdbwd:
            raise ValueError(f"RaftMatcher requires that fwdbwd option for RAFT is activated in the config!")
        flow_low, flow = self.raft(input_gray, ref_gray, iters=self.iters, test_mode=True, float_imgs01=True)
        N = input_gray.shape[0]

        # flow_fwd, flow_bwd = flow[0:N], flow[N:2*N]

        flow_fwd, flow_bwd = flow[0:N], flow[N:2*N]

        ref_col_warped, valid_ooim  = flow_warp_with_mask(ref_col, flow_fwd)
        ref_col_warped = 0.5 + (ref_col_warped - 0.5) * valid_ooim

        flowdelta, occlusion = est_occl_heu(flow_fwd, flow_bwd, self.occl_a1, self.occl_a2)
        flowdelta = torch.norm(flowdelta, p=2, dim=1, keepdim=True)

        occlusion = occlusion[:, None].float()
        confidence = valid_ooim[:,0:1] * (1.0 - occlusion)

        return ref_col_warped, flow, flow_low, flowdelta, confidence

