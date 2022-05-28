import unittest   
import torch                              
from parameterized import parameterized, parameterized_class  # parameterized-0.8.1
import os,sys 

_lib = os.path.join( os.path.dirname(os.path.abspath(__file__)),"../../")
if _lib not in sys.path:
    print(_lib)
    sys.path.insert(0, _lib)

import numpy as np
import matplotlib.pyplot as plt

from common_utils.feat_match.feat_matcher_utils import CostVolume, sample_with_grid_HWdydx, get_argmax_NHW_xy_last_np, \
                      get_argmax_NHW_xy_last_th, gen_2D_mg_xy_th




class Test(unittest.TestCase):
    def test_gen_2D_mg_xy_th(self):
        """Test the 2D Meshgrid provider Same o """
        H=4
        W=3
        mg = gen_2D_mg_xy_th(H,W)
        mgx = torch.tensor([[0, 1, 2,],
                            [0, 1, 2,],
                            [0, 1, 2,],
                            [0, 1, 2,]])
        mgy = torch.tensor([[0, 0, 0,],
                            [1, 1, 1,],
                            [2, 2, 2,],
                            [3, 3, 3,]])

        self.assertTrue( mg.shape == (1,2,H,W))  # N2HW
        self.assertTrue( type(mg) == torch.Tensor)
        self.assertTrue(np.allclose(mg[0,0] , mgx))
        self.assertTrue(np.allclose(mg[0,1] , mgy))

    @parameterized.expand([
       ("np", np.zeros,    get_argmax_NHW_xy_last_np),
       ("th", torch.zeros, get_argmax_NHW_xy_last_th),
    ])
    def test_get_argmax_NHW_xy_last(self, name, zeros, get_argmax_NHW_xy_last):
        """ Generic Unittest for numpy and pytorch """
        ix=2
        iy=3
        N = 2
        data = zeros([N,4,5])
        data[:,iy,ix] =1
        max_val, (coords) = get_argmax_NHW_xy_last(data)
        idx, idy = coords
        # print(max_val, (idx,idy))
        self.assertTrue( idx.shape == (N,))
        self.assertTrue( idy.shape == (N,))
        self.assertTrue( (idx==ix).sum() == N)
        self.assertTrue( (idy==iy).sum() == N)


    def test_sample_with_grid_HWdydx(self, rtol=1e-4):
        N,C,H,W = 3,2,5,7
        stride=None
        rx,ry = 1,3
        sx = 2*rx+1
        sy = 2*ry+1
        coords_ofs_xy = gen_2D_mg_xy_th(H,W).float()  
        data = torch.randn(N,C,H,W) 
        # data = -gen_2D_mg_xy_th(H,W).float()  # Use a grid as data for easier testing
        # data = data.repeat(N,1,1,1).contiguous()
        coords_ofs_xy = coords_ofs_xy.repeat(N,1,1,1).contiguous()        

        data_grid = sample_with_grid_HWdydx(data, coords_ofs_xy, rx=rx, ry=ry, stride=stride)
        #Test shape
        self.assertTrue(data_grid.shape == (N,C,H,W,sy,sx), f"Shape of sample_with_grid_HWdydx not as expected: {data_grid.shape}")

        #Test that center is truly centered
        cy, cx = sy//2, sx//2
        data_c = data_grid[..., cy, cx]
        self.assertTrue(torch.allclose (data_c, data, rtol=rtol) )

        ##########################################################################################
        # Test Integer shifts in different directions
        data_pad = torch.nn.functional.pad(data,(rx,rx,ry,ry), value=0)
        # Shift left:
        data_l = data_grid[..., cy, 0 ]
        data_comp = data_pad[:,:, 0+cy:H+cy, 0:W+0]
        self.assertTrue( torch.allclose(data_l,data_comp, rtol=rtol), f"integer shift left failed")

        # Shift top:
        data_t = data_grid[..., 0, cx ]
        data_comp = data_pad[:,:, 0+0:H+0, cx:W+cx]
        self.assertTrue( torch.allclose(data_t,data_comp, rtol=rtol), f"integer shift top failed")

        # Shift right:
        data_r = data_grid[..., cy, sx-1]
        data_comp = data_pad[:,:, 0+cy:H+cy, 0+sx-1:W+sx-1]
        self.assertTrue( torch.allclose(data_r,data_comp, rtol=rtol), f"integer shift top failed")

        # Shift bottom:
        data_b = data_grid[..., sy-1, cx]
        data_comp = data_pad[:,:, 0+sy-1:H+sy-1, 0+cx:W+cx]
        self.assertTrue( torch.allclose(data_b,data_comp, rtol=rtol), f"integer shift top failed")


    def test_sample_with_grid_HWdydx_stride(self):
        N,C,H,W = 3,2,5,7
        rx,ry, stride = 1,3,2
        sx = 2*rx+1
        sy = 2*ry+1
        cy, cx = sy//2, sx//2 # center
        coords_ofs_xy = gen_2D_mg_xy_th(H,W).float()  
        # data = torch.randn(N,C,H,W) 
        data = -1 -gen_2D_mg_xy_th(H,W).float()  # Use a grid as data for easier testing
        data = data.repeat(N,1,1,1).contiguous()
        coords_ofs_xy = coords_ofs_xy.repeat(N,1,1,1).contiguous()        

        data_grid = sample_with_grid_HWdydx(data, coords_ofs_xy, rx=rx, ry=ry, stride=stride)
        #Test shape
        self.assertTrue(data_grid.shape == (N,C,H,W,sy,sx), f"Shape of sample_with_grid_HWdydx not as expected: {data_grid.shape}")

        #Test that center is truly centered
        data_c = data_grid[..., cy, cx]
        self.assertTrue(torch.allclose (data_c, data))

        # Manual Test => data is a meshgrid check it manually on a few spots
        self.assertTrue( H//2 == 2 and W//2 ==3, f"test wrongly configured, manually setup for H==5,W==7")
        # Check x Direction of strides, 
        self.assertTrue(data[0,0, 2, 3 ] == -4)
        data_grid[0,0,2,3] == torch.tensor([[ 0.,  0.,  0.],
                                            [ 0.,  0.,  0.],
                                            [-2., -4., -6.],
                                            [-2., -4., -6.],
                                            [-2., -4., -6.],
                                            [ 0.,  0.,  0.],
                                            [ 0.,  0.,  0.]])
        # Check y direction of strides
        self.assertTrue(data[0,1,2,3] == -3)
        data_grid[0,1,2,3] == torch.tensor([[ 0.,  0.,  0.],
                                            [ 0.,  0.,  0.],
                                            [-1., -1., -1.],
                                            [-3., -3., -3.],
                                            [-5., -5., -5.],
                                            [ 0.,  0.,  0.],
                                            [ 0.,  0.,  0.]])



    def test_CostVolume_compute_CV(self):
        N,C,H,W = 3,1,6,8
        rx,ry, stride = 1,3,2
        sx = 2*rx+1
        sy = 2*ry+1
        cy, cx = sy//2, sx//2 # center
        ch,cw = H//2,W//2
        fmap1 = torch.zeros(N,C,H,W)
        fmap2 = torch.zeros(N,C,H,W)

        # Build and test a cost volume
        dx,dy = 2,1
        fmap1[:,:,ch,cw] = 1
        fmap2[:,:,ch+dy,cw+dx] = 1
        cv = CostVolume()
        corr_NH1W1_1_H2_W2 =  cv.compute_full_cv(fmap1, fmap2) # [N*H1*W1,1,H2,W2]
        corr_NH1W1H2W2 = corr_NH1W1_1_H2_W2.reshape(N,H,W,H,W)
        map2d = corr_NH1W1H2W2[0,ch,cw]
        self.assertTrue( map2d[ch+dy, cw+dx] == map2d.max() , msg="compute full cv failed")

    def test_CostVolume_get_wta_conf(self):
        N,C,H,W = 3,128,6,8
        rx,ry, stride = 1,3,2
        sx = 2*rx+1
        sy = 2*ry+1
        cy, cx = sy//2, sx//2 # center
        ch,cw = H//2,W//2
        dx,dy = 2,1
        ofs = int( max(abs(dx),abs(dy))+0.5)
        fmap = torch.rand(N,C,H+2*ofs,W+2*ofs)
        fmap1 = fmap[:,:, ofs:ofs+H, ofs:ofs+W].contiguous()
        fmap2 = fmap[:,:, ofs-dy:ofs-dy+H, ofs-dx:ofs-dx+W].contiguous()
        # fmap2 = torch.zeros(N,C,H,W)

        # Build and test a cost volume
        # fmap1[:,:,:,:] = 1
        # fmap2[:,:,ch+dy,cw+dx] = 1
        cv = CostVolume(fwdbwd=True)
        (flows_12_wta, flows_21_wta), (confs_12_wta, confs_21_wta), _ =    cv.get_wta_conf(fmap1, fmap2, cand_cnt=1)

        # Check Sizes
        self.assertTrue( (flows_12_wta[0].shape==(N,2,H,W)) , f"Flow field has wrong shape! Must be {[N,2,H,W]} but found {flows_12_wta[0].shape}!")
        self.assertTrue( (flows_21_wta[0].shape==(N,2,H,W)) , f"Flow field has wrong shape! Must be {[N,2,H,W]} but found {flows_21_wta[0].shape}!")
        self.assertTrue( (confs_12_wta[0].shape==(N,1,H,W)) , f"Confidence has wrong shape! Must be {[N,1,H,W]} but found {confs_12_wta[0].shape}!")
        self.assertTrue( (confs_21_wta[0].shape==(N,1,H,W)) , f"Confidence has wrong shape! Must be {[N,1,H,W]} but found {confs_21_wta[0].shape}!")

        # Crop center of flow (avoid boundary effects for test)
        # Flow field needed to get from fmap2->fmap1
        flow_21 = flows_21_wta[0][:,:,ofs:-ofs,ofs:-ofs]
        self.assertTrue(   flow_21[:,0].std() < 1e-5,  f"Center of flow field should have been identical values but differences detected! (x-values)")
        self.assertTrue(  flow_21[:,1].std() < 1e-5,  f"Center of flow field should have been identical values but differences detected! (y-values")
        self.assertTrue( (flow_21[0,0,0,0] + dx).abs() < 1e-5, f"Flow field wrong for flow_21 fmap2->fmap1")
        self.assertTrue( (flow_21[0,1,0,0] + dy).abs() < 1e-5, f"Flow field wrong for flow_21 fmap2->fmap1")

        flow_12 = flows_12_wta[0][:,:,ofs:-ofs,ofs:-ofs]
        self.assertTrue( flow_12[:,0].std() < 1e-5,  f"Center of flow field should have been identical values but differences detected! (x-values)")
        self.assertTrue( flow_12[:,1].std() < 1e-5,  f"Center of flow field should have been identical values but differences detected! (y-values")
        self.assertTrue( (-flow_12[0,0,0,0] + dx).abs() < 1e-5, f"Flow field wrong for flow_12 fmap1->fmap2")
        self.assertTrue( (-flow_12[0,1,0,0] + dy).abs() < 1e-5, f"Flow field wrong for flow_12 fmap1->fmap2")
        #TODO: Check this test

        # check matching confidences
        # fmap1 and fmap2 are identical but with a shift => highest matching value = for pixels with itself
        delta1 = confs_12_wta[0][:,:,ofs:-ofs,ofs:-ofs] - torch.sum(fmap1[:,:,ofs:-ofs,ofs:-ofs]**2, dim=1, keepdim=True)
        delta2 = confs_21_wta[0][:,:,ofs:-ofs,ofs:-ofs] - torch.sum(fmap2[:,:,ofs:-ofs,ofs:-ofs]**2, dim=1, keepdim=True)
        delta1 = delta1.abs().max().item()
        delta2 = delta2.abs().max().item()

        self.assertTrue(delta1 < 1e-4, f"Confidence value difference is too high! {delta1}")
        self.assertTrue(delta2 < 1e-4, f"Confidence value difference is too high! {delta2}")

    @parameterized.expand([   #  dx,dy
                                (-3, -3),
                                (-3, 0 ),
                                (-5, 5 ),
                                ( 3, 1 ),
                                ( 2, 2 ),
                                ( 0, 0 ),
    ])
    def test_CostVolume_refine(self, dx=1, dy=3):
        N,C,H,W = 3,64,32,16
        rx,ry, stride = 1,3,2
        sx = 2*rx+1
        sy = 2*ry+1
        cy, cx = sy//2, sx//2 # center
        ch,cw = H//2,W//2

        ofs = int( max(abs(dx),abs(dy))+0.5)*2
        Ho = H+2*ofs
        Wo = W+2*ofs
        fmap = torch.rand(N,C,Ho, Wo)
        fmap1 = fmap[:,:, ofs:ofs+H, ofs:ofs+W].contiguous()
        
        # Simulate best feature point by copying over 1 pixel to a dedicated position
        fmap2 = torch.zeros_like(fmap1)
        fmap2[:,:,ofs+dy:ofs+dy+1, ofs+dx:ofs+dx+1] = fmap1[:,:,ofs:ofs+1, ofs:ofs+1]

        feats1 = [fmap1]
        feats2 = [fmap2]
        for i in range(2):
            feats1.append (torch.nn.functional.avg_pool2d(feats1[-1], 2) )
            feats2.append (torch.nn.functional.avg_pool2d(feats2[-1], 2) )
        # Build and test a cost volume
        cv = CostVolume(fwdbwd=True, rx=2,ry=2)
        (flows_12_wta, flows_21_wta), (confs_12_wta, confs_21_wta), _ =    cv.get_wta_conf(feats1[-1], feats2[-1], cand_cnt=1, local_wdw=2 )

        flowcand, confref = cv.refine_cands(flows_12_wta, confs_12_wta, feats1, feats2)
        fx = flowcand[0][:,0,ofs:ofs+1, ofs:ofs+1]
        fy = flowcand[0][:,1,ofs:ofs+1, ofs:ofs+1]
        err_x = (fx-dx).abs().max()
        err_y = (fy-dy).abs().max()
        self.assertTrue( err_x < 1e-3, f"Sematic matching failed for x-axis by {err_x} for dx={dx},dy={dy}, but found fx{fx}, fy{fy}")
        self.assertTrue( err_y < 1e-3, f"Sematic matching failed for y-axis by {err_y} for dx={dx},dy={dy}, but found fx{fx}, fy{fy}")
        print(f"cv test passed for dx={dx},dy={dy}  {err_x},{err_y}")
    


if __name__ == '__main__':
    t = Test()
    unittest.main()
