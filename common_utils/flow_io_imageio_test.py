import unittest                                 
import os,sys 
import tempfile
from os.path import join, isfile
import numpy as np
import matplotlib.pyplot as plt
from pandas import value_counts


_cdir =  os.path.dirname(os.path.abspath(__file__))+"/"
print(_cdir)
if _cdir not in sys.path:
    print(_cdir)
    sys.path.insert(0, _cdir)

from flow_io_imageio import writeDeltaFlowUint16, readDeltaFlowUint16, load_Kitti_png_flow, save_Kitti_png_flow
from flow_io_cv2 import readFlowKITTI, writeFlowKITTI

class Test(unittest.TestCase):
    def test_writeDeltaFlowUint16(self):
        # create a file on the default ubuntu ram drive /dev/shm/

        delta = np.clip( np.random.randn( 128,1340)*100, 0,255)
        with tempfile.TemporaryDirectory() as tmp:
            fn = (join(tmp,"delta_flow_export_test.png"))
            writeDeltaFlowUint16(fn, delta)
            delta_in = readDeltaFlowUint16(fn)
            os.remove(fn)

        abs_delta = np.abs(delta - delta_in).max()
        abs_delta_ok = abs_delta <= 1/64

        self.assertTrue(abs_delta_ok, msg=f"delta = {abs_delta}") 
        print(f"Flow Delta file write + reload equal?: {abs_delta_ok}")


    def test_save_and_reload_Kitti_flow(self):
        H,W = 128,1340
        # H,W = 12,14
        flow = np.clip( np.random.randn( H,W, 2)*512, -511,511)
        mask = np.random.choice((0,1),size=(H,W,1))
        flow = flow * mask
        flow_cmbd = np.concatenate([flow,mask], axis=-1)
        with tempfile.TemporaryDirectory() as tmp:
            fn = (join(tmp,"flow_export_test.png"))
            save_Kitti_png_flow(fn, flow_cmbd)
            flow_in, mask_in = load_Kitti_png_flow(fn, cat=False)
            os.remove(fn)
            
        flow_delta = np.abs(flow - flow_in).max()
        mask_delta = np.abs(mask - mask_in).max()
        flow_delta_ok = flow_delta <= 1/64
        mask_delta_ok = mask_delta == 0

        # fig, axes = plt.subplots(2,2,num=1,clear=True)
        # vmin = min(flow_in.min(), flow.min())
        # vmax = max(flow_in.max(), flow.max())
        # axes[0,0].imshow(mask[...,0])
        # axes[1,0].imshow(mask_in[...,0])
        # axes[0,1].imshow(flow[...,0],vmin=vmin,vmax=vmax)
        # axes[1,1].imshow(flow_in[...,0],vmin=vmin,vmax=vmax)
        # plt.show()

        print(f"Flow file write + reload delta {flow_delta} - ok?: {flow_delta_ok}")
        print(f"Flow file write + reload delta {mask_delta} - ok?: {mask_delta_ok}")
        self.assertTrue(flow_delta_ok, msg=f"delta = {flow_delta}") 
        self.assertTrue(mask_delta_ok, msg=f"delta = {mask_delta}")

    def test_save_and_reload_cv2_Kitti_flow(self):
        H,W = 128,1340
        # H,W = 12,14
        flow = np.clip( np.random.randn( H,W, 2)*100, -511,512)
        mask = np.random.choice((0,1),size=(H,W,1))
        flow = flow * mask
        flow_cmbd = np.concatenate([flow,mask], axis=-1)
        with tempfile.TemporaryDirectory() as tmp:
            fn = (join(tmp,"flow_export_test.png"))
            save_Kitti_png_flow(fn, flow_cmbd)
            flow_in_cv2, mask_in_cv2 = readFlowKITTI(fn)
            os.remove(fn)
            
        mask_in_cv2 = mask_in_cv2[...,None]
        flow_delta = np.abs(flow - flow_in_cv2).max()
        mask_delta = np.abs(mask - mask_in_cv2).max()
        flow_delta_ok = flow_delta <= 1/64
        mask_delta_ok = mask_delta == 0

        print(f"Flow file write + reload delta {flow_delta} - ok?: {flow_delta_ok}")
        print(f"Flow file write + reload delta {mask_delta} - ok?: {mask_delta_ok}")

        # fig, axes = plt.subplots(2,2,num=1,clear=True)
        # vmin = min(flow_in_cv2.min(), flow.min())
        # vmax = max(flow_in_cv2.max(), flow.max())
        # axes[0,0].imshow(mask[...,0])
        # axes[1,0].imshow(mask_in_cv2[...,0])
        # axes[0,1].imshow(flow[...,0],vmin=vmin,vmax=vmax)
        # axes[1,1].imshow(flow_in_cv2[...,0],vmin=vmin,vmax=vmax)
        # plt.show()

        self.assertTrue(flow_delta_ok, msg=f"delta = {flow_delta}") 
        self.assertTrue(mask_delta_ok, msg=f"delta = {mask_delta}")

    def test_load_Kitti_flow(self):
        fn = _cdir + 'data/00000__00001_04d_flowvec.png'
        if not isfile(fn):
            raise ValueError(f"File Not found {fn}")
        flow_vec_cv2, mask_cv2 = readFlowKITTI(fn)
        flow_vec_imageio, mask_imageio = load_Kitti_png_flow(fn, cat=False)
        
        mask_cv2 = mask_cv2[...,None]
        flow_ok = np.allclose(flow_vec_cv2, flow_vec_imageio)
        mask_ok = np.allclose(mask_cv2, mask_imageio)
        self.assertTrue(flow_ok, msg=f"Flow Vector loaded with imageio is not equal to cv2 version!")
        self.assertTrue(mask_ok, msg=f"Flow Mask loaded with imageio is not equal to cv2 version!")
        print(f"Flow Comparison with CV2 ok?: {flow_ok and mask_ok}")




if __name__ == '__main__':
    t = Test()
    # t.test_writeDeltaFlowUint16()
    unittest.main()
