import sys
import os
import shutil
from os.path import join
from imageio import imread, imsave, mimsave
import numpy as np
import random
import torch
import torch.optim as optim
from model import VideoRecolourer

from common_utils.utils import UINT16MAX, UINT8MAX
from common_utils.utils import log_config
from common_utils.flow_utils import flow2uint16restoresim, est_occl_heu, flow_warp_with_mask
from common_utils.data_transformations import propagate_old_masks 

from argparse import Namespace
from RAFT_custom.core.raft import RAFT
from RAFT_custom.download_models import download_raft_models
from model_matcher import FeatureMatcher

from typing import Dict, Optional, List, Tuple


class LVVCP_Combined_inference(torch.nn.Module):
    def __init__(self, config, logger):
        super(LVVCP_Combined_inference, self).__init__()        
        self.config = config
        self.logger = logger

        # Setup which color proposals to use based on config
        self._useCandsFlow:bool= True      # Default setup always uses Flow
        self._useCandsGlobal:bool = False  #
        self._useCandsLocal:bool = False   #
        self._setup_cand_config(config)
        
        self._simulate_file_quantizations : bool = True # Historic, to better reproduce research results
        self._freeze_bugs: bool = True # Historic, freeze bugs discovered during code refactoring to better reproduce research results

        # Setup Placeholders for the sub modules
        self._raft_model :torch.nn.Module = None
        self._fuse_and_refine :torch.nn.Module = None
        self._feature_matcher_local:torch.nn.Module = None
        self._feature_matcher_global:torch.nn.Module = None

    def set_S(self, S):
        return self._fuse_and_refine.set_S(S)

    def get_S(self):
        return self._fuse_and_refine.get_S()
        

    def _setup_cand_config(self, config):
        """ Setup which color proposal candidates to be used based on config
            Changes: 
            self._useCandsFlow
            self._useCandsGlobal
            self._useCandsLocal
        """
        if  not ( ('cands' in config['MaskPredictor']['config']) and (type(config['D']['config']['init_image']) == list) ):
            raise ValueError (f"Invalid or outdated config, must contain a ['MaskPredictor']['config'] and a ['D']['config']['init_image']\n {config} ")

        settings2attribs = {'flow':'_useCandsFlow', 'glob':'_useCandsGlobal', 'loc':'_useCandsLocal'}
        for key, var in settings2attribs.items():
            if  key  in config['MaskPredictor']['config']['cands']:
                setattr(self, var, True)
                self.logger.info(f"Found {key} in config['MaskPredictor']['config']['cands'] => turn on proposal generation for {key}={getattr(self, var)}")
        return


    def restore_research_code_artifacts(self):
        """ Restores some uneccesary artifacts from original research code for better reproducibility of values
        """
        self._simulate_file_quantizations  = True # Historic, to better reproduce research results
        self._freeze_bugs = True # Historic, freeze bugs discovered during code refactoring to better reproduce research results

        # Freeze known bugs for reproducibility
        self.logger.info(f"Setting up matching networks for local and global matching") 
        if self._freeze_bugs:
            # Feature matching backbones were previously stored in a List, hence the eval signal for
            #    batchnorm got lost, and this sub-part stayed in training mode (default)
            self._feature_matcher_local.train()   # deactivate batchnorm updates
            self._feature_matcher_global.train()  # deactivate batchnorm updates

            self._feature_matcher_local.feat_matcher.cv.legacy_mode = True # OOImage warping 0 point used 0, but should be 0.5=gray
            self._feature_matcher_global.feat_matcher.cv.legacy_mode = True # OOImage warping 0 point used 0, but should be 0.5=gray
            self.logger.info(f"FB MODE ON") 
        else:
            self._feature_matcher_local.eval()   # deactivate batchnorm updates
            self._feature_matcher_global.eval()  # deactivate batchnorm updates

            self._feature_matcher_local.feat_matcher.cv.legacy_mode = False # 
            self._feature_matcher_global.feat_matcher.cv.legacy_mode = False # 
            self.logger.info(f"FB MODE OFF") 


    def load_sub_models(self, config, save_path:str, SfixedOnly:int=-1):
        """ Reloads the different Sub-models into the combined model, using config file
        """
        # Load RAFT
        raft_args = Namespace(**config['RaftFlow'])
        self._raft_model = RAFT(raft_args)

        download_raft_models()

        # md5sum   models/raft-sintel.pth     cc69e5da1f38673ab10d1849859ebe91
        raft_ckpt = {k.replace('module.',''):v for k,v in torch.load(raft_args.model, map_location='cpu').items() if k.startswith('module') }
        self._raft_model.load_state_dict(raft_ckpt)
        self._raft_iters = config['RaftFlow']['iters']
        self._raft_occl_a1 = config['RaftFlow']['occl_a1']
        self._raft_occl_a2 = config['RaftFlow']['occl_a2']
        self.logger.info(f"Restoring RAFT Optial flow from {raft_args.model}, using {raft_args.iters} iters") 

        # Load local and global matchers:
        self._feature_matcher_local  = FeatureMatcher(config['feat_match_local'], self.logger)
        self._feature_matcher_global = FeatureMatcher(config['feat_match_global'], self.logger)


        # Load & Refinement Model
        self._fuse_and_refine = VideoRecolourer(config, self.logger)
        # load pretrained weights if desired
        if config['init_pretrained']:
            self.logger.info(f"Restoring model from f'{config['init_pretrained']}'")
            # Sanity checks:
            if not os.path.isfile(config['init_pretrained']):
                raise ValueError( f"could not find  config['init_pretrained']  '{config['init_pretrained']}'" )
            checkpoint = torch.load(config['init_pretrained'], map_location='cpu') #first load to CPU RAM
            if not 'version' in checkpoint:
                raise ValueError(f"entry version not found, possibly incompatible checkpoint")
            if not checkpoint['version'] == 'p2':
                 raise ValueError(f"Wrong version of checkpoint! Expected v2 but found:'{checkpoint['version']}'")
            # Making a backup in output folder
            fn_modelbak = os.path.join(save_path, f"model_used.ckpt")
            shutil.copyfile(config['init_pretrained'], fn_modelbak)

            if 'epoch' in checkpoint:
                self.logger.info(f"Restoring from epoch:'{checkpoint['epoch']}'")
                                   
            # load the model state dict
            self._fuse_and_refine.load_state_dict(checkpoint['model'])
            if SfixedOnly >= 0:
                stages =  int(SfixedOnly)
                self._fuse_and_refine.set_separate_stages_count(stages)
                self.logger.info(f"SfixedOnly: stages count is now independent from S, removing dependence of T from iterations:  S={checkpoint['S']},  stages={stages}")


    def forward(self, data:Dict[str, torch.Tensor], data_prev:Dict[str, torch.Tensor], global_refs:Dict[str, List[torch.Tensor]], get_res=True):
        """ LVVCP - Combined model for inference

        Receives

            data = {
                'i0c':  i_prev_col,                  # previous coloured image  [N3HW] (Lab)
                'i0g3': i_prev_gray3,                # previous image gray channel  [N3HW] (3xduplicated)

                'i1L':  i_curr_luminance,            # current image luminace channel [N1HW]
                'i1g3': i_curr_gray3,                # current image gray channel (3x) [N3HW] (3xduplicated)              
            }
            
            data_prev : a dictionary of results from previous frame, or empty dict if new sequence

                          
            global_refs = {
                'ic':  i_refs_col,          # List of Global refernces - color Lab [N3HW]
                'ig3': i_refs_gray3,        # List of Global refernces - gray [N3HW]               
                'im1': i_refs_mask,         # [Optional] List of Global refernces - Alpha masks [N3HW]  
            }   
        """
        if self._fuse_and_refine is self._raft_model is self._feature_matcher_local is self._feature_matcher_global is None:
            raise ValueError(f"Seems like pre-trained models have not been loaded - please call 'load_sub_models' first ")

        # Data sanity check
        for key, C in {'i0c':3,'i0g3':3,'i1L':1,'i1g3':3}.items():
            if key not in data:
                raise ValueError(f" missing key:'{key}' in data dictionary!")
            if len(data[key].shape) != 4 and data[key].shape[1] != C:
                raise ValueError(f" Wrong dimensionality of data[{key}]: should be [N,{C},H,W] but is {data[key].shape}")
        
        for key, C in {'ic':3,'ig3':3}.items():
            if key not in global_refs:
                raise ValueError(f" missing key:'{key}' in global_refs dictionary!")
            for i_gr, gr in enumerate(global_refs[key]):
                if len(gr.shape) != 4 and gr.shape[1] != C:
                    raise ValueError(f" Wrong dimensionality of global_refs[{key}]: should be [N,{C},H,W] but is {global_refs[key].shape}")
        
        # Compute Color proposal candidates
        if self._useCandsFlow:
            self.computeFlowCands(data)

        if self._useCandsLocal:
            self.computeLocalCands(data)

        if self._useCandsGlobal:
            self.computeGlobalCands(data, global_refs)


        # propagete previous data 
        if data_prev is not None:
            propagate_old_masks(data, data_prev)

        x_colorized, res = self._fuse_and_refine(data, get_res=True)

        return x_colorized, res



    def computeFlowCands(self, data:Dict[str, torch.Tensor]):
        """ Computes Color proposal candidates based on optical flow, and heurisitics

            requires: 
                data = {
                    'i1g3': i_curr_gray3,       # current image gray channel (3x) [N3HW] (3xduplicated)
                    'i0g3': i_prev_gray3,       # previous image gray channel  [N3HW] (3xduplicated)
                }
            
            adds the following keys to data dict:
                'i0cw'             # Warped previous color image (Lab)
                'fdelta10'         # An optical flow delta
                'flow10'           # High resolution optical flow
                'flow_low_fwd'     # low resolution optical flow (fwd)
                'flow_low_bwd'     # low resolution optical flow (bwd)
                'matched10_hard'   # A heurisitic on matched pixels

        """
        self.logger.info(f"Matching:  Starting Flow Estimation") 

        for key in ['i0g3', 'i1g3']:
            if key not in data:
                raise ValueError(f"Input Error: did not find '{key}' in data keys {[data.keys()]}")
            if data[key].shape[1] != 3:
                raise ValueError(f"Input shape for {key} must be [N3HW] (color or duplicated gray) but is {data[key].shape}")
                
        #1.a compute Flow in Fwd & Bwd direction
        flow_low, flow_up = self._raft_model(data['i1g3'], data['i0g3'], iters=self._raft_iters, test_mode=True, float_imgs01=True)
        #1.b) compute Flow features
        flow_up_fwd, flow_up_bwd   = flow_up[0:1],  flow_up[1:2]
        flow_low_fwd, flow_low_bwd = flow_low[0:1],flow_low[1:2]
        #1.c) compute flow heuristics
        flowdelta_fwd, occl_fwd = est_occl_heu(flow_up_fwd, flow_up_bwd, self._raft_occl_a1, self._raft_occl_a2)
        flowdelta_fwd  = torch.norm(flowdelta_fwd,p=2,dim=1,keepdim=True)                
        self.logger.info(f"Matching: Flow & Heuristics Done") 

        #1.d) compute warped flow 
        i_prev_warp_col, valid_ooim  = flow_warp_with_mask( data['i0c'], flow_up_fwd)
        i_prev_warp_col =  0.5 + (i_prev_warp_col-0.5) * valid_ooim  # image data is between [0,1] with 0.5 being the 0 point
        self.logger.info(f"Matching: Flow Warping done ") 

        # Adding Flow Matches to data structure
        data['i0cw']           = i_prev_warp_col
        data['fdelta10']       = flowdelta_fwd  
        data['flow10']         = flow_up_fwd  
        data['flow_low_fwd']     = flow_low_fwd  
        data['flow_low_bwd']     = flow_low_bwd  
        data['matched10_hard'] = 1.0 - occl_fwd[:,None].float()  
        # data['matched10_soft'] = torch.zeros_like(data['matched10_hard'])
        data['matched10_soft'] =  data['matched10_hard'] 

        # simulate uint16 file saving & restoring (including value clipping), as in training:
        if self._simulate_file_quantizations:
            data['fdelta10'] = torch.clamp(256.*data['fdelta10'],0,2**16-1)/256. 
            data['matched10_hard'] = torch.clamp(255.*data['matched10_hard'],0,255)/255. 
            data['matched10_soft'] = torch.clamp(255.*data['matched10_soft'],0,255)/255. 
            data['flow10'] = flow2uint16restoresim(data['flow10'])

        self.logger.info(f"Matching:  Flow Estimation Done") 

    def computeLocalCands(self, data:Dict[str, torch.Tensor]):
        """ Computes Color proposal candidates based on LOCAL feature matching and adds result to data dictionary

            requires: 
                data = {
                    'i1g3': i_curr_gray3,       # current image gray channel (3x) [N3HW] (3xduplicated)   

                    'i0g3': i_prev_gray3,       # previous image gray channel  [N3HW] (3xduplicated)
                    'i0c':  i_prev_col,         # previous coloured image  [N3HW] (Lab)

                    'flow_low_bwd',              #  optical flow (low res) bwd  [N2HW]
                    'flow_low_fwd',              #  optical flow (low res) fwd  [N2HW]
                }
            
            adds the following keys to data dict:
                'colcands1_loc'        # Cands      [N,C,H,W,K] => [N,K,C,H,W]
                'colcands1_best_loc'   # Best Cand  [N,C,H,W]
                'conf_best_loc'        # Confidence [N,1,H,W]
        """
        self.logger.info(f"Matching: Local ") 
        loc_colcand, loc_best_cand_conf, loc_best_cand_dist_conf = self._feature_matcher_local(
                                                                            input_gray = data['i1g3'],
                                                                            references_gray = [data['i0g3']],
                                                                            references_col = [data['i0c']],
                                                                            motion_flows_fwd=[data['flow_low_fwd']],
                                                                            motion_flows_bwd=[data['flow_low_bwd']])

        # simulate file saving & restoring (including clipping):
        if self._simulate_file_quantizations:
            loc_best_cand_conf      = torch.clamp(UINT16MAX * loc_best_cand_conf     ,0,UINT16MAX).to(dtype=torch.int32).float()/UINT16MAX
            loc_best_cand_dist_conf = torch.clamp(UINT16MAX * loc_best_cand_dist_conf,0,UINT16MAX).to(dtype=torch.int32).float()/UINT16MAX
        loc_best_cand_dist_conf = 1.0 - loc_best_cand_dist_conf

        # Add Local Matches to datastructure
        data['colcands1_loc']       = loc_colcand              # Cands      [N,C,H,W,K] => [N,K,C,H,W]
        data['colcands1_best_loc']  = loc_best_cand_conf       # Best Cand  [N,C,H,W]
        data['conf_best_loc']       = loc_best_cand_dist_conf  # Confidence ['N,1,H,W]

        self.logger.info(f"Matching: Global - Refinement of Matching - Done") 


    def computeGlobalCands(self, data:Dict[str, torch.Tensor], global_refs:Dict[str, List[torch.Tensor]]):        
        """ Computes Color proposal candidates based on LOCAL feature matching and adds result to data dictionary

            requires: 
                data = {
                    'i1g3': i_curr_gray3,       # current image gray channel (3x) [N3HW] (3xduplicated) 
                }
                
                global_refs = {
                    'ic':  i_refs_col,          # List of Global refernces - color Lab [N3HW]
                    'ig3': i_refs_gray3,        # List of Global refernces - gray [N3HW]               
                    'im1': i_refs_mask,         # [Optional] List of Global refernces - Alpha masks [N3HW]  
                }

            adds to data dict:
                'colcands1_glob'       # Cands      [N,C,H,W,K] => [N,K,C,H,W]
                'colcands1_best_glob'  # Best Cand  [N,C,H,W]
                'conf_best_glob'       # Confidence [N,1,H,W]
        """
        self.logger.info(f"Matching: Global ")
        glob_colcand, glob_best_cand_conf, glob_best_cand_dist_conf = self._feature_matcher_global(
                                                                            input_gray = data['i1g3'],
                                                                            references_gray = global_refs['ig3'],
                                                                            references_col = global_refs['ic'],
                                                                            references_masks = global_refs['im1']
                                                                            )

        # simulate file saving & restoring (including clipping):
        if self._simulate_file_quantizations:
            glob_best_cand_conf       = torch.clamp(UINT16MAX * glob_best_cand_conf     ,0,UINT16MAX).to(dtype=torch.int32).float()/UINT16MAX
            glob_best_cand_dist_conf  = torch.clamp(UINT16MAX * glob_best_cand_dist_conf,0,UINT16MAX).to(dtype=torch.int32).float()/UINT16MAX
        # Also Transforms Uncertaintiy Into confidence:  Conf = (1-Uncertainty)
        glob_best_cand_dist_conf  = 1.0 - glob_best_cand_dist_conf

        # Add Global Matches to datastructure
        data['colcands1_glob']      = glob_colcand              # Cands      [N,C,H,W,K] => [N,K,C,H,W]
        data['colcands1_best_glob'] = glob_best_cand_conf       # Best Cand  [N,C,H,W]
        data['conf_best_glob']      = glob_best_cand_dist_conf  # Confidence [N,1,H,W]
        self.logger.info(f"Matching: Global - Refinement of Matching - Done") 
    
