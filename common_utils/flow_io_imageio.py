# This file provides various methods to save & reload optical flow in a Kitti similar format.
# i.e. flow is discretized as 16-bit values and stored as 16-bit lossless png to save memory.
#  flow_discrete =  (flow_cont * 2**6 + 2**15) = (flow_cont + 2**6) * 2**9
#  The range of the flow is therefore ~ +/- 511 pixel, with 1/64 resolution
#  Values larger than this range will be set to 0 (saves memory) and marked as invalid

import imageio
import numpy as np 

UINT16MAX = (2**16)-1   # 65535
KITTI_FLOW_MUL = 2**6   # 64
KITTI_FLOW_OFS = 2**15  # 32768


def writeDeltaFlowUint16(filename, deltaflow, compression=6):
    """ Saves absolute flow delta, max val = 256, max prec 1/256 """
    if len(deltaflow.shape) != 2:
        raise ValueError(f"Wrong shape, expected [H,W] but is {deltaflow.shape}")
    deltaflow = 256.0 * np.abs(deltaflow)
    deltaflow = np.clip(deltaflow,0,2**16-1)
    deltaflow = deltaflow.astype(np.uint16)
    imageio.imsave (filename, deltaflow ,format ="PNG-FI", compression=compression) 

def readDeltaFlowUint16(filename):
    """ Loads absolute flow delta, max val = 256, max prec 1/256 """
    deltaflow = imageio.imread (filename ,format ="PNG-FI")
    if deltaflow.dtype != np.uint16 : f"FlowDelta PNG images are 1 channel a 16-bit images, but this is {deltaflow.dtype}, {deltaflow.shape}"
    if len(deltaflow.shape) != 2 :    f"FlowDelta PNG images are 1 channel a 16-bit images, but this is {deltaflow.dtype}, {deltaflow.shape}"

    if deltaflow is None:
        raise ValueError(f"Error could not read :{filename}")
    if deltaflow.dtype != np.uint16:
        raise ValueError(f"Wrong dataformat for file:{filename}, expected uint16 but is {deltaflow.dtype}")
    if len(deltaflow.shape) != 2:
        raise ValueError(f"Wrong shape for file:{filename}, expected [H,W] but is {deltaflow.shape}")
    deltaflow = deltaflow.astype(np.float) /256.0
    return deltaflow


def save_Kitti_png_flow(fname, flow, compression=6, clamp_nMaskInvalid=True):
  """
  saves a Kitti 16-bit 3channel RGB (non standard RGB!) and converts it back to flow information
  compression [1..9] 1 lowest compression (largest files but fastest), 9 highest compression (smaller files but slower)
  """

  if len(flow.shape) != 3 : raise ValueError(f"Expected [HW2] shape for flow but is {flow.shape}")
  if flow.shape[-1] not in [2,3]  : raise ValueError(f"Expected [HW2] or [HW3] shape for flow but is {flow.shape}")
  
  valid_mask_in = None
  if flow.shape[-1] == 3:
    valid_mask_in = flow[...,2] != 0 

  flow = flow * KITTI_FLOW_MUL + KITTI_FLOW_OFS

  # Check if flow fits data_format 
  valid_mask = (flow[...,0:2] >= 0) & (flow[...,0:2]  <= UINT16MAX)
  valid_mask = valid_mask[...,0] & valid_mask[...,1]
  if valid_mask_in is not None:
    valid_mask = valid_mask & valid_mask_in
  if clamp_nMaskInvalid:
    valid_mask = valid_mask   # implicit type conversion that works for numpy and pytorch
    invalid_mask = ~valid_mask

    # mask out invalid flow vectors that do not fit data format
    flow[invalid_mask,0] = np.full_like(flow[invalid_mask,0],KITTI_FLOW_OFS)
    flow[invalid_mask,1] = np.full_like(flow[invalid_mask,1],KITTI_FLOW_OFS)
  else:
    if np.any(valid_mask):
      Warning.warn(f"Values have been clipped for {fname}, Maximum Flow in KITTI format is +/-511pix use maskInvalid if you want to exclude these pixels instead. However this results in a sparse flow")
    # Clamp the Flow if it goes out of bounce
    valid_mask =  np.zeros_like(flow[...,0:1])
    flow =  np.clip(flow, 0, UINT16MAX)
    
  flow = flow.astype(np.uint16)
  valid_mask = np.clip(valid_mask.astype(np.uint16),0,1)
  flow_rgb = np.concatenate( [flow[...,0:2], valid_mask[...,None]],axis=-1)  # [u,v,valid]

  flow_rgb_int = flow_rgb.astype(np.uint16)
  # Use free-image library - many libraries have troubles with 16-Bit RGB pngs
  imageio.imsave (fname, flow_rgb_int, format="PNG-FI", compression=compression)
  return


def load_Kitti_png_flow(fname, cat=False):
  """  
  loads a Kitti SIMILAR 16-bit 3channel RGB (non standard RGB!) and converts it back to flow information
  """

  flow_rgb = imageio.imread (fname ,format ="PNG-FI")
  assert flow_rgb.dtype == np.uint16, "Kitti Flow PNG images are 3 channel a 16-bit images"
  flow_raw = flow_rgb.astype(np.float32)
  flow2D =  (flow_raw[...,0:2] - KITTI_FLOW_OFS)/KITTI_FLOW_MUL
  valid  =  (flow_raw[...,2:3] )
  if cat:
    ret = np.concatenate([flow2D, valid],axis=-1)
  else:
    ret = (flow2D, valid)
  return ret 



def save_Kitti_SIMILAR_png_flow(fname,flow,occ):
  """
  saves a Kitti SIMILAR 16-bit 3channel RGB (non standard RGB!) and converts it back to flow information
  
  The only difference is that instead of a valid bit in the B channel there is the occlusion calculated from backmatch is saved as continuous value
  """
  
  assert flow.shape[-1] == 2 , "Expected HW2 for flow"
  assert flow.ndim  == 3 , "Expected HW2 for flow"
  assert occ.ndim == 2 , "Expected HW for occlusions"

  flow_rg = flow * KITTI_FLOW_MUL + KITTI_FLOW_OFS
  occ_b = occ * 256  # only positive values => 
  
  flow_rgb = np.concatenate( [flow_rg, occ_b[...,np.newaxis]],axis=-1)

  flow_rgb_int = flow_rgb.astype(np.uint16)

  imageio.imsave (fname,flow_rgb_int ,format ="PNG-FI")
  return


def load_Kitti_SIMILAR__png_flow(fname):
  """
  loads a Kitti SIMILAR 16-bit 3channel RGB (non standard RGB!) and converts it back to flow information
  
  The only difference is that instead of a valid bit in the B channel there is the occlusion calculated from backmatch is saved as continuous value
  """
  
  flow_rgb = imageio.imread (fname ,format ="PNG-FI")
  assert flow_rgb.dtype == np.uint16, "Kitti Flow PNG images are 3 channel a 16-bit images"
  flow_raw = flow_rgb.astype(np.float32)
  flow2D =  (flow_raw[...,0:2] - KITTI_FLOW_OFS)/KITTI_FLOW_MUL
  valid  =  (flow_raw[...,2:3] )/256.0
  return (flow2D, valid)

