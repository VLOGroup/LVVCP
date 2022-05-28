# This file provides various methods to save & reload optical flow in a Kitti similar format.
# i.e. flow is discretized as 16-bit values and stored as 16-bit lossless png to save memory.
#  flow_discrete =  (flow_cont * 2**6 + 2**15) = (flow_cont + 2**6) * 2**9
#  The range of the flow is therefore ~ +/- 511 pixel, with 1/64 resolution
#  Values larger than this range will be set to 0 (saves memory) and marked as invalid

import cv2
import numpy as np 

UINT16MAX = (2**16)-1
KITTI_FLOW_MUL = 2**6   # 64
KITTI_FLOW_OFS = 2**15


def writeDeltaFlowUint16(filename, deltaflow):
    """ Saves absolute flow delta, max val = 256, max prec 1/256 """
    if len(deltaflow.shape) != 2:
        raise ValueError(f"Wrong shape, expected [H,W] but is {deltaflow.shape}")
    deltaflow = 256.0 * np.abs(deltaflow)
    deltaflow = np.clip(deltaflow,0,2**16-1)
    deltaflow = deltaflow.astype(np.uint16)
    return cv2.imwrite(filename, deltaflow)    

def readDeltaFlowUint16(filename):
    """ Loads absolute flow delta, max val = 256, max prec 1/256 """
    deltaflow = cv2.imread(filename, cv2.IMREAD_ANYDEPTH|cv2.IMREAD_GRAYSCALE)
    if deltaflow is None:
        raise ValueError(f"Error could not read :{filename}")
    if deltaflow.dtype != np.uint16:
        raise ValueError(f"Wrong dataformat for file:{filename}, expected uint16 but is {deltaflow.dtype}")
    if len(deltaflow.shape) != 2:
        raise ValueError(f"Wrong shape for file:{filename}, expected [H,W] but is {deltaflow.shape}")
    deltaflow =  deltaflow.astype(np.float) /256.0
    return deltaflow


def load_Kitti_png_flow(filename, cat=True):
  flow,valid  = readFlowKITTI(filename)
############################################
# from RAFT
def readFlowKITTI(filename):
    flow = cv2.imread(filename, cv2.IMREAD_ANYDEPTH|cv2.IMREAD_COLOR)
    flow = flow[:,:,::-1]
    flow, valid = flow[:, :, :2], flow[:, :, 2]
    flow = flow.astype(np.float32)
    flow = (flow - 2**15) / 64.0
    return flow, valid

def readDispKITTI(filename):
    disp = cv2.imread(filename, cv2.IMREAD_ANYDEPTH) / 256.0
    valid = disp > 0.0
    flow = np.stack([-disp, np.zeros_like(disp)], -1)
    return flow, valid


def writeFlowKITTI(filename, uv):
    uv = 64.0 * uv + 2**15
    valid = np.ones([uv.shape[0], uv.shape[1], 1])
    uv = np.concatenate([uv, valid], axis=-1).astype(np.uint16)
    cv2.imwrite(filename, uv[..., ::-1])