import os
from os.path import join
import torch
import numpy as np
from imageio import imread, imwrite, imsave, mimsave
from common_utils.color_conversion_pytorch import rgb2lab_normalized_NCHW, lab2rgb_normalized_NCHW
import multiprocessing

from common_utils.utils import pt2np, np2uin8img, np2uint16img, check_makedirs, get_logger, psnr, deepcopy2cpu
from common_utils.color_conversion_pytorch import rgb2lab_normalized_NCHW, lab2rgb_normalized_NCHW
from common_utils.torch_script_logging import TorchScriptLogger, log_info, log_warning

from RAFT_custom.core.utils.utils import InputPadder

import multiprocessing
import atexit
from typing import List


def get_file_saver(vis_type, save_anim=True, save_path="", color_mode="Lab", minimal=False, logger=None):
    if vis_type=="MP":
        return SavedColorizedIF( save_anim=save_anim, save_path=save_path, color_mode=color_mode, minimal=minimal, logger=logger)
    elif vis_type=="std":
        return SaveColorized( save_anim=save_anim, save_path=save_path, color_mode=color_mode, minimal=minimal, logger=logger)
    elif vis_type=="none":
         return SaveColorized_NOOutput(save_anim=save_anim, save_path=save_path, color_mode=color_mode, minimal=minimal, logger=logger)
    else:
        raise ValueError(f"Wrong setting for vis_type:'{vis_type}' must be 'MP' or 'std' ")



class SaveColorizedMP(multiprocessing.Process):
    def __init__(self, data_queue, save_anim=True, save_path="out_val/", color_mode="Lab", minimal=False, logger=None):
        multiprocessing.Process.__init__(self)
        self._shutdown = multiprocessing.Event()

        self.data_queue = data_queue
        self.save_path = save_path
        self.minimal=minimal
        self.logger = logger
        self.save_anim = save_anim
        self.color_mode=color_mode

        self._save_colorized = SaveColorized(save_anim=save_anim, save_path=save_path, minimal=minimal, color_mode=color_mode)

    def run(self):
        proc_name = self.name
        while not self._shutdown.is_set():
            try:
                data = self.data_queue.get()
                log_info("MP_Saver: before file saving")
                self.save(data)
                self.data_queue.task_done()
                log_info("MP_Saver: after file saving")
            except:
                self.data_queue.task_done()
                with open(self.save_path + "/error_log_vis.txt", "w") as fp:
                    fp.write(f"MP_Saver: proces {proc_name} crashed")
                print("MP_Saver: file saving process crashed")
                log_info("MP_Saver: file saving process crashed")
                raise

    def save(self,data):
        """
        This function will be called with a copy of the data by a separate process to speed up visualization
        """
        log_info(f"MP_Saver: start MP file saving ")

        if 'save_path' in data:
            save_path = data['save_path']
            if len(save_path)> 0:
                save_path=save_path + "/"
            del data['save_path']
            self.save_path = save_path


        if "padder" in data:
            self._save_colorized.set_padder(data['padder'])
            return 

        if "save_anim" in data:
            print("            self._save_colorized.save_anim()")
            self._save_colorized.save_anim()
            print("            self._save_colorized.save_anim() done")
            return 

        self._save_colorized(**data)

    def shutdown(self):
        self._shutdown.set()
        log_info("MP_Saver: Shuttdown called")

class SavedColorizedIF(object):
    writer = None
    def __init__(self, save_anim=True, save_path="out_val/", color_mode="Lab", minimal=False, logger=None):
        # Establish communication queues
        self.save_path = save_path
        self._save_anim = save_anim
        self.color_mode=color_mode
        self.minimal = minimal
        self.logger = logger
        self.data_queue = multiprocessing.JoinableQueue()
        self._start_mp_worker()

    def set_padder(self,padder):
        self.data_queue.put({'padder':padder})

    def _start_mp_worker(self):
        # Generate a MultiProcessing Visualizer
        def terminate_process(MP_Saver):
            print("MP_Saver:Progamm exitting: shutting down visualizer")
            MP_Saver.terminate()
        self.MP_Saver = SaveColorizedMP(self.data_queue, save_anim=self._save_anim, save_path=self.save_path, color_mode=self.color_mode, minimal=self.minimal, logger=self.logger)
        self.MP_Saver.start()
        atexit.register(terminate_process, self.MP_Saver)

    def save_anim(self):
        self.data_queue.put({'save_anim':True})


    def __call__(self, i1c_gt, i2c_gt, i2c_est, m10_est, init=False, seq_name="", i2c_est_init=None, out_file_str=None, idb=0):
        """ put the data onto the queue for the MP saver  to take it over """
        i1c_gt = deepcopy2cpu(i1c_gt, idb=idb)
        i2c_gt = deepcopy2cpu(i2c_gt, idb=idb)
        i2c_est = deepcopy2cpu(i2c_est, idb=idb)
        m10_est = deepcopy2cpu(m10_est, idb=idb)
        if i2c_est_init is not None:
            i2c_est_init = deepcopy2cpu(i2c_est_init, idb=idb)

        if (not self.MP_Saver.is_alive()) or (self.data_queue.qsize() > 500):
            self.logger.error("Multiprocess Worker seems to have died!")
            raise AssertionError("Multiprocess Worker seems to have died!")

        data = {
            'i1c_gt':       i1c_gt,
            'i2c_gt':       i2c_gt,
            'i2c_est':      i2c_est,
            'm10_est':      m10_est,
            'init':         init,
            'seq_name':     seq_name,
            'i2c_est_init': i2c_est_init,
            'out_file_str': out_file_str,
        }
        self.data_queue.put(data)

        log_info(f"MP_Saver: data put on queue, queue-size:{self.data_queue.qsize()}")

    def is_busy(self):
        return self.data_queue.qsize() > 0
    
    def shutdown(self, wait=True):
        self.MP_Saver.shutdown()
        if wait:
            self.MP_Saver.data_queue.join()
        log_info(f"MP_Saver: Shutdown finished")


class SaveColorized(object):
    def __init__(self, save_anim=True, save_path="", color_mode="Lab", minimal=False, logger=None):
        self._save_anim_val = save_anim
        self._data_est =[]
        self._data_cmbd =[]
        self._data_est_init =[]
        self.save_path = save_path
        self.savepath_with_seq = save_path
        self.padder = None
        self.minimal = minimal
        self.logger = logger
        if color_mode in ['Lab', 'RGBLateLab']:  # LateLab returns RGB from dataloader but converts it to Lab on the GPU:
            self.isLab = True
        elif  color_mode == 'RGB':
            self.isLab = False
        else:
            assert False, f"wrong color mode"
    def set_padder(self,padder):
        self.padder = padder
    def img2RGBUint8(self, img_th, scale=255, idb=0):
        if self.padder is not None:
            # Undo padding if a padder is present
            img_th = self.padder.unpad(img_th)

        if self.isLab and img_th.shape[1] == 3:
            img_th = lab2rgb_normalized_NCHW(img_th)
        img_rgb = torch.clamp(img_th[idb]*scale,0,255).to(dtype=torch.uint8).cpu().numpy().transpose(1,2,0)
        if img_rgb.shape[-1] == 1:
            img_rgb = np.ascontiguousarray(img_rgb)
            img_rgb = np.tile(img_rgb,(1,1,3))
        return img_rgb
    def __call__(self, i1c_gt, i2c_gt, i2c_est, m10_est, init=False, seq_name="", i2c_est_init=None, out_file_str=None):
        save_path = self.save_path
        if seq_name:
            save_path = join(self.save_path, seq_name)
            self.savepath_with_seq = save_path

        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        if i2c_est_init is None:
            i2c_est_init = torch.zeros_like(i2c_est)
        i2c_est_init_np = self.img2RGBUint8(i2c_est_init)
        i2c_est_np = self.img2RGBUint8(i2c_est)
        i2c_gt_np  = self.img2RGBUint8(i2c_gt)
        i1c_gt_np  = self.img2RGBUint8(i1c_gt)
        z = np.zeros_like(i2c_gt_np)
        o = np.ones_like(i2c_gt_np)*255

        if self.minimal:
            # output minimal data then return immediatly
            if init:
                imsave(join(save_path,f"00000000_init_frame.png"), i1c_gt_np)
            imsave(join(save_path,f"{out_file_str}.png"), i2c_est_np)
            if self._save_anim_val:
                self._data_est.append( i2c_est_np)
            return


        if init:
            self._data_est =[]
            self._data_cmbd =[]
            cmbd_np = np.concatenate([np.concatenate( [i1c_gt_np,   i1c_gt_np],axis=1),
                                      np.concatenate( [o        ,   z        ],axis=1)],axis=0)
            imsave(join(save_path,f"frames_01_colorized_000_init_frame.png"), i1c_gt_np)
            imsave(join(save_path,f"frames_00_pre_refinement_col_000_init_frame.png"), i1c_gt_np)
            imsave(join(save_path,f"frames_02_combined_000_init_frame.png") , cmbd_np)

            self._data_est =[i1c_gt_np]
            self._data_est_init =[i1c_gt_np]
            self._data_cmbd =[cmbd_np]


        delta =  (i2c_gt[0:1] - i2c_est[0:1]).abs().sum(dim=1,keepdim=True)
        delta_np = self.img2RGBUint8(delta, scale=255)
        m10_est_np = self.img2RGBUint8(m10_est)
        cmbd_np = np.concatenate([ np.concatenate( [i2c_gt_np,  i2c_est_np],axis=1),
                                   np.concatenate( [m10_est_np, delta_np ],axis=1)],axis=0)
                                   

        imsave(join(save_path,f"frames_00_pre_refinement_col_{out_file_str}.png"), i2c_est_init_np)
        imsave(join(save_path,f"frames_01_colorized_{out_file_str}.png"), i2c_est_np)
        imsave(join(save_path,f"frames_02_combined_{out_file_str}.png" ), cmbd_np)
        imsave(join(save_path,f"frames_03_delta_{out_file_str}.png"    ), delta_np[...,0])
        imsave(join(save_path,f"frames_04_mask_{out_file_str}.png"     ), m10_est_np[...,0])
        imsave(join(save_path,f"frames_05_gt_{out_file_str}.png"       ), i2c_gt_np)

        if self._save_anim_val:
            self._data_est.append( i2c_est_np)
            self._data_cmbd.append( cmbd_np )
            self._data_est_init.append(i2c_est_init_np)

    def save_anim(self):
        if not self.save_anim:
            return
        try:
            kwargs = { 'macro_block_size': None }
            kwargs = { }
            if len(self._data_est     ): mimsave(join(self.savepath_with_seq,f"frames_000_colorized_anim_fast.mp4")               , self.pad_to_div16(self._data_est), fps=10, **kwargs)
            if len(self._data_cmbd    ): mimsave(join(self.savepath_with_seq,f"frames_000_colorized_anim_fast_cmbd.mp4")          , self.pad_to_div16(self._data_cmbd), fps=10, **kwargs)
            if len(self._data_est     ): mimsave(join(self.savepath_with_seq,f"frames_000_colorized_anim_slow.mp4")               , self.pad_to_div16(self._data_est), fps=1, **kwargs)
            if len(self._data_cmbd    ): mimsave(join(self.savepath_with_seq,f"frames_000_colorized_anim_slow_cmbd.mp4")          , self.pad_to_div16(self._data_cmbd), fps=1, **kwargs)
            if len(self._data_est_init): mimsave(join(self.savepath_with_seq,f"frames_000_pre_refinement_col_anim_fast_cmbd.mp4") , self.pad_to_div16(self._data_est_init), fps=10, **kwargs)
            if len(self._data_est_init): mimsave(join(self.savepath_with_seq,f"frames_000_pre_refinement_col_slow_cmbd.mp4")      , self.pad_to_div16(self._data_est_init), fps=1, **kwargs)
        except BaseException as err:
            print(f"Something went wrong when generating animation video {err}")

        self._data_est =[]
        self._data_cmbd =[]
        self._data_est_init=[]

    def pad_to_div16(self, data_lst=List[np.ndarray]):
        if data_lst:
            H,W,C = data_lst[0].shape
            hpad = (16 - H%16) if H%16 else 0 
            wpad = (16 - W%16) if W%16 else 0 
            data_lst = [ np.pad(data, ([0,hpad],[0,wpad],[0,0])) for data in data_lst]
        return data_lst



    def is_busy(self):
        return False
    
    def shutdown(self):
        pass

class SaveColorized_NOOutput(object):
    def __init__(self, *args, **kwargs):
        pass
    def __call__(self, *args, **kwargs):
        pass
    def save_anim(self, *args, **kwargs):
        pass
    def is_busy(self):
        return False
    def set_padder(self,padder):
        pass
    def shutdown(self):
        pass
