import torch
import os, sys
import numpy as np
import cv2
import imageio 

_lib =  os.path.dirname(os.path.abspath(__file__))+"/"
print(_lib)
if _lib not in sys.path:
    print(_lib)
    sys.path.insert(0, _lib)

import unittest

from flow_utils import flow2uint16restoresim, flow2uint16, uint16flow2flow

class TestCases(unittest.TestCase):
    def test_uint16flow2flow(self):
        flow_uint16 = np.arange(2**16).reshape(2**8,2**8).astype(np.uint16)
        flow_vec = uint16flow2flow(flow_uint16)
        flow_roundtrip = flow2uint16(flow_vec)        
        is_same = np.allclose(flow_uint16, flow_roundtrip)
        self.assertTrue(is_same, msg=f"Flow uint16 conversion did not yield same result as before")

    def test_flow2uint16(self):
        flow = np.arange(2**16).reshape(2**8,2**8)/64 - 512  # -512.. (+512-1/64)
        print(flow.min(), flow.max())
        flow_uint16 = flow2uint16(flow)
        flow_roundtrip = uint16flow2flow(flow_uint16) 
        delta_max = np.abs(flow - flow_roundtrip).max()
        is_same = (delta_max <= 1/64)
        self.assertTrue(is_same, msg=f"Flow uint16 conversion did not yield same result as before, max_delta={delta_max}")





if __name__ == '__main__':
    unittest.main()


    print('done')