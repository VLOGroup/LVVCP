"""
This Module generates global matches between grayscale images using pre-trained segmentation models
from the standard pytorch torchvision modelzoo.

https://pytorch.org/docs/stable/torchvision/models.html#semantic-segmentation

Image Normalization required:
  As with image classification models, all pre-trained models expect input images normalized in the same way.
  The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406]
  and std = [0.229, 0.224, 0.225]. They have been trained on images resized such that their minimum size is 520.
"""

import os.path as path
import numpy as np
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt

import logging 
def log_info(msg):
    logging.getLogger("main-logger").info(msg)

# For loacl image normalization
from skimage import exposure
from skimage.morphology import disk
from skimage.filters import rank

from .feat_matcher_utils import  CostVolume
                        
# Various Models for feature extraction
from torchvision import transforms
import torchvision.models as models
# Modified Resent Classifiers
from .resnet import resnet101

from typing import List, Optional, Dict, Union
from .RAFT_extractor import BasicEncoder, SmallEncoder

__all__ = ['get_global_feat_matcher', ]

def normalize_gray_img(img):
    assert img.max() > 1, f"at least a few pixels need to have larger values than 1, image expected to be 0..255"""

    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    # Means and standard deviations from imagenet used to train the classifiers
    pt_mean = 0.449 # = np.mean([0.485, 0.456, 0.406])
    pt_std  = 0.226 # = np.mean([0.229, 0.224, 0.225])

    img = img / 255.0
    return (img - pt_mean) / pt_std


def get_global_feat_matcher(backbone, fwdbwd, config):
    # VGG type backbones
    if backbone == "Classification_VGG16":
        return FeatMatcher_ClassVGG16_2LevelOnly(fwdbwd=fwdbwd, config=config)
    elif backbone == "Classification_VGG16Multi":
        return FeatMatcher_ClassVGG16Multi(fwdbwd=fwdbwd, config=config)
    elif backbone == "Classification_VGG16bn":
        return FeatMatcher_ClassVGG16bnMulti(fwdbwd=fwdbwd, config=config)
    # ResNet 101 type backbones
    elif backbone == "Deeplabv3_ResNet101_multi":
        return FeatMatcher_DeeplabResNet101_multi(fwdbwd=fwdbwd, config=config)
    elif backbone == "Classification_Resnet101_Multi":
        return FeatMatcher_ClassResNet101_Multi(fwdbwd=fwdbwd, config=config)
    elif backbone == "FeatMatcher_RAFT_Basic":
        return FeatMatcher_RAFT_Basic(fwdbwd=fwdbwd, config=config)         
    else:
        raise ValueError(f"unkown backbone '{backbone}'")

class FeatureNormalizer(torch.nn.Module):
    def __init__(self,config):
        super(FeatureNormalizer, self).__init__()
        self.config = config
        self.logger = logging.getLogger("main-logger")

        self._instance_norm = False
        self._NCC_norm = False
        self._NCC_0mean = False
        if 'featnorm' in config:
            known_normalizations = set(['instance_norm','NCC_norm','NCC_0mean'])
            if not known_normalizations.issuperset(config['featnorm']):
                raise ValueError(f"unkown configuration for feature normalization found in {config['featnorm']}"
                                 f"most likely wrong key: {known_normalizations.union(config['featnorm']).difference(known_normalizations)}")
            if 'instance_norm' in config['featnorm']:
                self._instance_norm = True
            if 'NCC_norm' in config['featnorm']:
                self._NCC_norm = True
            if 'NCC_0mean' in config['featnorm']:
                self._NCC_0mean = True
            self.logger.warning(f"FeatMatcher {self}")
        else:
            self.logger.warning(f"FeatMatcher {self}: featnorm: Not using any normalization - most likely a configuration error")

    def forward(self, tens_lst: List[torch.Tensor]):

        for i, tens in enumerate(tens_lst):
            
            # Manually implemented Instance Norm
            #  \tilde{f} =  (f - mean(f, [H,W])) / std(f, [H,W])
            #  Normalize Features to be roughly equally expensive
            if self._instance_norm:
                tens = tens - tens.mean(dim=(2,3),keepdim=True)
                tens = tens / torch.clamp_min(torch.std(tens,dim=(2,3),keepdim=True), 1e-6) # Feature Normalization

            if self._NCC_0mean:
                # for legacy compatibility - is actually superfluous and was close to 0 most of the time anyway
                tens = tens - tens.mean(dim=(1,2,3),keepdim=True)

            # Normalize Images to be centered unit vectors, since we want cosine distance (normalize cross correllation)
            #  NCC =  <f1,f2> / (|f1| |f2|) =  < f1/|f1| , f2/|f2| >  
            #    => normalize features independently first before computing NCC
            if self._NCC_norm:
                tens = tens / torch.clamp_min(torch.norm(tens,p=2,dim=1,keepdim=True), 1e-6)
            tens_lst [i] = tens


        return tens_lst    

    def __repr__(self):
        return f"FeatureNormalizer(config={{'instance_norm':{self._instance_norm}, 'NCC_norm':{self._NCC_norm}, 'NCC_0mean':{self._NCC_0mean} }})"
        
        
class FeatMatcher(torch.nn.Module):
    def __init__(self, fwdbwd:bool, config):
        super(FeatMatcher, self).__init__()
        self.fwdbwd = fwdbwd
        self.config = config
        self.config_str = str(config)

        self.logger = logging.getLogger("main-logger")

        self.normalizer = FeatureNormalizer(config)

        if 'old_CV_norm' in config and config['old_CV_norm']:
            raise NotImplementedError(f"old_CV_norm is not supported Any More!!!")

        if 'old_local_wdw' in config and config['old_local_wdw']:
            raise NotImplementedError(f"old_local_wdw is not supported Any More!!!")
        
        # Coarse matching:
        self.local_wdw_sz = 0
        if 'wdw_sz' in config and config['wdw_sz']:
            self.local_wdw_sz = int(config['wdw_sz'])
            self.logger.info(f"Matcher uses wdw_sz:{self.local_wdw_sz}")
        
        self.useFlowOfs:bool=False
        if 'useFlowOfs' in config and config['useFlowOfs']:
            self.useFlowOfs=bool(config['useFlowOfs'])
            self.logger.info("using Flow Offset in Matcher activated")
            if not self.local_wdw_sz:
                raise ValueError(f"useFlowOfs is set while local_wdw_sz is deactivated! This is not allowed! 'useFlowOfs'={ config['useFlowOfs']},  'wdw_sz'={self.local_wdw_sz}")

        self.cand_cnt = 8
        if 'cand_cnt' in config and config['cand_cnt']:
            self.cand_cnt = config['cand_cnt']
            self.logger.info(f"Matcher - generating {self.cand_cnt} candidates")

        # Match Refinement:
        self.rx = 2
        self.ry = 2 
        if 'dxy' in config:
            self.rx = config['dxy']
            self.ry = config['dxy']

        sample_cv_op='PyTorch'
        if 'sample_cv_op' in config and config['sample_cv_op']:
            sample_cv_op = config['sample_cv_op']
        feat_upsample='nearest'
        if 'upsample' in config and config['upsample']:
            feat_upsample = config['upsample']

        self.multi = bool(config['multi'])
        self.cmb_confs = bool(config['cmb_confs'])

        self.cv = CostVolume(use_autocast=False, fwdbwd=self.fwdbwd, sample_cv_op_type=sample_cv_op, feat_upsample=feat_upsample, rx=self.rx, ry=self.ry, multi=self.multi, cmb_confs=self.cmb_confs)

    def img2feats(self, I):
        """ Computes images from the features """
        I_norm = normalize_gray_img(I)
        feat_lst = self.feats(I_norm)
        # Do a normalization - helps for correlation based features
        feat_lst = self.normalizer(feat_lst)
        # Start Matching part
        return feat_lst[-1].contiguous(), feat_lst

    def get_wta_matches_from_imgs(self, I1_gray:torch.Tensor, I2_gray:torch.Tensor, flow12_low: Optional[torch.Tensor]=None, flow21_low: Optional[torch.Tensor]=None):
        """ Interface for backward compatibility """
        # Generate features - Uses Instance Normalization + l2 norm => yields unit vectors for later dot product.
        f1_low, feats_1 = self.img2feats(I1_gray)
        f2_low, feats_2 = self.img2feats(I2_gray)

        # Perform a WTA (winner takes all) matching on the coarsest scale
        wta_flow =  self.get_wta_matches(f1_low, f2_low, flow12_low, flow21_low)
        return wta_flow, feats_1, feats_2 # backward compatibility

    
    def get_wta_matches(self, f1_gray_low:torch.Tensor, f2_gray_low:torch.Tensor, flow12_low: Optional[torch.Tensor]=None, flow21_low: Optional[torch.Tensor]=None):
        """ Perform a winner takes all matching on a coarse (low) resolution
        """
        if not torch.jit.is_scripting():
            assert type(f1_gray_low) == torch.Tensor, f"Error occured, features should be a tensor but is {type(f1_gray_low)}"
            assert type(f2_gray_low) == torch.Tensor, f"Error occured, features should be a tensor but is {type(f2_gray_low)}"
        assert f1_gray_low.shape == f2_gray_low.shape, f"Error occured, f1_gray_low.shape! f2_gray_low.shape  |  {f1_gray_low.shape}!={f2_gray_low.shape}"
        N,C,H,W = f1_gray_low.shape

        if self.useFlowOfs and ((flow12_low is None) or (flow21_low is None)):
            raise ValueError(f"'useFlowOfs'={ self.useFlowOfs} but no Flow is provided! {self.config_str}")
        if not self.useFlowOfs and ((flow12_low is not None) or (flow21_low is not None)):
            raise ValueError(f"'useFlowOfs'={ self.useFlowOfs} but Flow is provided! {self.config_str}")

        # Three cases for matching:
        #  - Global (No Flow, no window size)
        #  - Local with Flow offset => local window + flow
        #  - Local without Flow => local window, no flow, just around same position
        if (flow12_low is not None) and (flow21_low is not None):
            # Using Flow => safety checks: flow is here, and window is set
            if (flow12_low.shape[-2:] != f1_gray_low.shape[-2:]) or (flow12_low.shape[0] != N)  : 
                raise ValueError(f"Flow f1_gray_low:{f1_gray_low.shape}  vs. flow01:{flow12_low.shape}")            
            if (flow21_low.shape[-2:] != f1_gray_low.shape[-2:]) or (flow21_low.shape[0] != N)  : 
                raise ValueError(f"Flow f1_gray_low:{f1_gray_low.shape}  vs. flow01:{flow21_low.shape}") 
            if not self.local_wdw_sz:
                raise ValueError(f"'useFlowOfs' is used but no wdw_sz={self.local_wdw_sz} was provided => this renders flow offset useless! Please set 'useFlowOfs' to False")

            #Std case for local matching (with Flow)
            (flows_12_wta, flows_21_wta), (confs_12_wta, confs_21_wta), _ = self.cv.get_wta_conf(f1_gray_low, f2_gray_low, local_wdw=self.local_wdw_sz, flow12_low=flow12_low, flow21_low=flow21_low, cand_cnt=self.cand_cnt)
            # self.logger.info("Matcher: Using Flow + wdw_sz={self.local_wdw_sz}")
        else:
            # Alternative case for local or global matching (without flow)
            (flows_12_wta, flows_21_wta), (confs_12_wta, confs_21_wta), _ = self.cv.get_wta_conf(f1_gray_low, f2_gray_low, local_wdw=self.local_wdw_sz, flow12_low=None, flow21_low=None, cand_cnt=self.cand_cnt)
            # self.logger.info("Matcher: Only using wdw_sz={self.local_wdw_sz}")
        
        return (flows_12_wta, flows_21_wta), (confs_12_wta, confs_21_wta), (None, None)

    def forward(self, I1_gray, I2_gray, I2_col, flow12_low: Optional[torch.Tensor]=None, flow21_low: Optional[torch.Tensor]=None):
        """
        Converts images to features, and extracts colors on matched positions
        """
        # Generate features - Uses Instance Normalization + l2 norm => yields unit vectors for later dot product.
        f1_low, feats_1 = self.img2feats(I1_gray)
        f2_low, feats_2 = self.img2feats(I2_gray)

        # Perform a WTA (winner takes all) matching on the coarsest scale
        (flows_12_wta, flows_21_wta), (confs_12_wta, confs_21_wta), _ = self.get_wta_matches(f1_low, f2_low, flow12_low, flow21_low)
        # Refine the coarse matching
        flowcand_12, confref_12 = self.cv.refine_cands( flows_12_wta, confs_12_wta, feats_1, feats_2)
        # extract colors at matched positions
        colcands_12, confref_12 = self.cv.sample_colcands(flowcand_12, confref_12, reference_col=I2_col)

        return (colcands_12, confref_12, flowcand_12)



class FeatMatcher_ClassVGG16_2LevelOnly(FeatMatcher):
    def __init__(self, fwdbwd, config):
        super(FeatMatcher_ClassVGG16_2LevelOnly,self).__init__(fwdbwd, config)
        
        VGG_FEAT_POS = 15 # ds = 4
        VGG_FEAT_POS = 17 # ds = 8 (incl cnn => 256 feats)
        VGG_FEAT_POS = 22 # ds = 8 (incl cnn => 517 feats)
        vgg16_model=models.vgg16(pretrained=True)
        self._backbones = torch.nn.ModuleList([ vgg16_model.features[:1],
                                                vgg16_model.features[1:VGG_FEAT_POS] ])
    def feats(self, img):
        """Convert an image tensor to a list of feature tensors at different scales """
        feats = [img]
        for bb in self._backbones:
            feats += [bb(feats[-1])]
        feats = feats[1:]
        return feats


class FeatMatcher_ClassVGG16Multi(FeatMatcher):
    def __init__(self, fwdbwd, config):
        super(FeatMatcher_ClassVGG16Multi,self).__init__(fwdbwd, config)
        
        VGG_DS1_FEAT64  = 0  # ds = 1 (incl cnn &BN => 64  feats)
        VGG_DS2_FEAT128 = 5  # ds = 2 (incl cnn &BN => 128 feats)
        VGG_DS4_FEAT256 = 10 # ds = 4 (incl cnn &BN => 256 feats)
        VGG_DS8_FEAT512 = 17 # ds = 8 (incl cnn &BN => 512 feats)
        vgg16_model=models.vgg16(pretrained=True)
        self._backbones = torch.nn.ModuleList([ 
                            vgg16_model.features[0                  : VGG_DS1_FEAT64  +1],
                            vgg16_model.features[VGG_DS1_FEAT64  +1 : VGG_DS2_FEAT128 +1],
                            vgg16_model.features[VGG_DS2_FEAT128 +1 : VGG_DS4_FEAT256 +1],
                            vgg16_model.features[VGG_DS4_FEAT256 +1 : VGG_DS8_FEAT512 +1],
                            ])
    def feats(self, img):
        """Convert an image tensor to a list of feature tensors at different scales """
        feats = [img]
        for bb in self._backbones:
            feats += [bb(feats[-1])]
        feats = feats[1:]
        return feats

class FeatMatcher_ClassVGG16bnMulti(FeatMatcher):
    def __init__(self, fwdbwd, config):
        super(FeatMatcher_ClassVGG16bnMulti,self).__init__(fwdbwd, config)
        
        # Feature positions for VGG16bn, always first layer with that amount of feats, after bn
        VGG_DS1_FEAT64  = 1  # ds = 1 (incl cnn &BN => 64  feats)
        VGG_DS2_FEAT128 = 8  # ds = 2 (incl cnn &BN => 128 feats)
        VGG_DS4_FEAT256 = 15 # ds = 4 (incl cnn &BN => 256 feats)
        VGG_DS8_FEAT512 = 25 # ds = 8 (incl cnn &BN => 512 feats)
        vgg16bn_model=models.vgg16_bn(pretrained=True)
        self._backbones = torch.nn.ModuleList([
                            vgg16bn_model.features[0                  : VGG_DS1_FEAT64  +1],
                            vgg16bn_model.features[VGG_DS1_FEAT64  +1 : VGG_DS2_FEAT128 +1],
                            vgg16bn_model.features[VGG_DS2_FEAT128 +1 : VGG_DS4_FEAT256 +1],
                            vgg16bn_model.features[VGG_DS4_FEAT256 +1 : VGG_DS8_FEAT512 +1],
                        ])
    def feats(self, img):
        """Convert an image tensor to a list of feature tensors at different scales """
        feats = [img]
        for bb in self._backbones:
            feats += [bb(feats[-1])]
        feats = feats[1:]
        
        return feats

class FeatMatcher_RAFT_Basic(FeatMatcher):
    def __init__(self, fwdbwd, config):
        super(FeatMatcher_RAFT_Basic,self).__init__(fwdbwd, config)

        self._backbone = BasicEncoder(output_dim=256, norm_fn='instance', dropout=0.0)    

        #restore backbone from RAFT 
        if not 'model_file' in config:
            cdir = path.split(__file__)[0]
            model_file = path.join(cdir,'models/raft-sintel.pth')
        else:
            model_file = config['model_file']
        log_info(f"RAFT backbone loading {model_file}")
        ckpt = torch.load(model_file)
        ckpt_backbone = {k.replace('module.fnet.',''):v for k,v in ckpt.items() if k.startswith('module.fnet')}
        self._backbone.load_state_dict(ckpt_backbone)
        self._backbone = self._backbone

        # Make a copy of 1st conv and re-do it without stride to get a full res image
        c2d = self._backbone.conv1
        self._conv1_no_stride = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=True)
        self._conv1_no_stride.load_state_dict(c2d.state_dict())

    def feats(self, img):
        feats_lst =  self._backbone(img, get_intermediate=True)
        feats_hr = self._conv1_no_stride(img)
        feats_lst = [feats_hr] + feats_lst
        feats_lst[0].shape # first conv layer + bn    ds=1
        feats_lst[1].shape #   end of 1st deep layer before ReLU  ds=2
        feats_lst[2].shape #   end of 2nd deep layer before ReLU  ds=4
        feats_lst[3].shape #   end of 3rd deep layer before ReLU  ds=8
        return feats_lst
        


        
class FeatMatcher_ClassResNet101_Multi(FeatMatcher):
    def __init__(self, fwdbwd, config):
        super(FeatMatcher_ClassResNet101_Multi,self).__init__(fwdbwd, config)
        self._backbone = resnet101(get_intermediate=True)
        # Make a copy of 1st conv and re-do it without stride to get a full res image
        c2d = self._backbone.conv1
        self._conv1_no_stride = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
        self._conv1_no_stride.load_state_dict(c2d.state_dict())

    def feats(self, img):
        class_out, feats_lst =  self._backbone(img)
        feats_hr = self._conv1_no_stride(img)
        feats_lst[0].shape # first conv layer + bn    ds=2
        feats_lst[1].shape #   end of 1st deep layer before ReLU  ds=4
        feats_lst[2].shape #   end of 2nd deep layer before ReLU  ds=8
        feats_lst = [feats_hr] + feats_lst[0:3] 
        return feats_lst[0:4]
        
class FeatMatcher_DeeplabResNet101_multi(FeatMatcher):
    def __init__(self, fwdbwd, config):
        super(FeatMatcher_DeeplabResNet101_multi,self).__init__(fwdbwd, config)
        segmentation_backbone = models.segmentation.deeplabv3_resnet101(pretrained=True).backbone
        self._backbone = resnet101(get_intermediate=True)
        incompatible_keys = self._backbone.load_state_dict( segmentation_backbone.state_dict() ,strict=False)
        if incompatible_keys.missing_keys != ['fc.weight', 'fc.bias']:
            raise ValueError(f"Seems like something changed i nthe pretrained FCNResNet101 - unable to load weights correctly")

        self._backbone  = self._backbone

        c2d = self._backbone.conv1
        self._conv1_no_stride = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
        self._conv1_no_stride.load_state_dict(c2d.state_dict())

    def feats(self, img):
        class_out, feats_lst =  self._backbone(img)
        feats_hr = self._conv1_no_stride(img)
        feats_lst[0].shape # first conv layer + bn    ds=2
        feats_lst[1].shape #   end of 1st deep layer before ReLU  ds=4
        feats_lst[2].shape #   end of 2nd deep layer before ReLU  ds=8
        feats_lst = [feats_hr] + feats_lst[0:3] 
        return feats_lst[0:4]
