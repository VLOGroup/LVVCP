import sys
import os
import shutil
from os.path import join
import time
from ast import literal_eval
import argparse
import imageio
from imageio import imread, imsave, mimsave
import numpy as np
import random
import torch
import torch.optim as optim
from model import VideoRecolourer
import pandas as pd
from ast import literal_eval
from copy import deepcopy
import logging

from torch.utils.tensorboard import SummaryWriter
import matplotlib
if 'MATPLOTLIBBACKEND' in os.environ and os.environ['MATPLOTLIBBACKEND'] == 'Agg':
    print('Running Matplotlib in NO-GUI mode, for shell execution only')
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common_utils.utils import bcolors, UINT16MAX, UINT8MAX
from common_utils.output_generation import SaveColorized, get_file_saver
from common_utils.utils import check_makedirs, get_logger, log_config, psnr, DictAverager, load_cfg_log, join_dicts
from common_utils.utils_demo import load_img, load_and_clean_fielname_txt, filnamestr_to_refs, CandOptionsParser, get_data_for_next_run

import json

from typing import Dict, Optional, List, Tuple

from model_inference_combined import LVVCP_Combined_inference


#%%
def load_json_config(args):
    """ loads a default config, and overrides default values if requested"""

    # Check the pre-trained model path
    model_path = args.load_path
    if not os.path.isfile(model_path):
        raise ValueError(f"Could not find model.pth under path '{model_path}'")

    # Check the model config.json file
    cfg_dir = os.path.split(model_path)[0]
    cfg_json_path = os.path.join(cfg_dir,"config.json")
    if not os.path.isfile(cfg_json_path):
        raise ValueError(f"Did not find a config.json next to the model.pth file. Expected at: {cfg_json_path}")
    with open(cfg_json_path) as fp:
        config = json.load(fp)
    # Get model filename
    model_name = os.path.splitext(os.path.split(model_path)[1])[0]

    # Setup the config to poitn to the model.pth file
    config['init_pretrained'] = model_path

    # Setup and check Dataset path
    if args.dataset_path:
        config['data']['path'] = args.dataset_path
        print(f"Using External Provided dataset path: {args.dataset_path}")
    if not os.path.isdir(config['data']['path']):
        raise ValueError(f"Did not find a Dataset directory under: {config['data']['path']}")

    # Setup and check that a dataset list is available
    if args.list_val:
        config['data']['list_val'  ] = args.list_val
        print(f"Using External Provided list_val: {args.list_val}")
    if not os.path.isfile(config['data']['list_val']):
        raise ValueError(f"Did not find a dataset txt file under: {config['data']['list_val'  ]}")

    if args.verbose !=-1:
        assert  args.verbose in [0,1], f"verbose must be [-1,0,1] where -1 means ignore use config, 0 is on, 1 is off but is {args.verbose}"
        config['train']['verbose'] = args.verbose

    if config['data']['color_mode'] not in ['Lab','RGB','RGBLateLab']:
        raise ValueError(f"color_mode:{config['data']['color_mode']} not supported")

    return config, model_name


def setup_output_paths(args, config, model_name=""):
    """ Builds a standardized output path based on the config and input arguments
    """

    time_prefix = time.strftime('%Y_%m%d_%H%M', time.localtime())
    SfixedPrefix = "Sfixed_{args.SfixedOnly}" if args.SfixedOnly > 0 else ""
    cand_count_postfix = f"_candcnt{args.cand_count:02}" if args.cand_count >= 0 else ""
    postfix = f"{model_name}_{args.out_postfix}{cand_count_postfix}"
    # save_path = os.path.join(args.out_path, f"{time_prefix}_{SfixedPrefix}_{config['prefix']}bs{config['data']['bs_train']}{postfix}/")

    save_path = f"{SfixedPrefix}_{config['prefix']}bs{config['data']['bs_train']}{postfix}/"
    if args.skip_automatic_prefix:
        save_path = f"vis_" + save_path
    else:
        save_path = f"vis_" + time_prefix + save_path
    save_path = os.path.join(args.out_path, save_path)

    os.makedirs(save_path, exist_ok=True)
    return save_path

def main(args):
    ###################################
    # Load config from pre-trained model (looking for a model.json next to the model.pth file):
    config, model_name = load_json_config(args)

    # Build an output path:
    save_path = setup_output_paths(args, config, model_name)

    # Setup the logger:
    logger = get_logger(args, filepath=save_path )
    logger.info(f"LVVCP_inference Demo:")

    # Build a file saver class with either simple direct output or faster multi-processing output
    file_saver = get_file_saver(args.filesaver_type, save_anim=True, save_path=save_path, minimal=args.minimal_output, color_mode="Lab")

    # Load list of files to process
    file_list = load_and_clean_fielname_txt(args.list_val)
    logger.info(f"Running Demo on Filelist '{args.list_val}'")



    logger.info(f"LVVCP build and load model from pre-trained state:")
    # Build the full model and load pre-trained parameters
    model = LVVCP_Combined_inference(config, logger)
    model.load_sub_models(config, save_path,  SfixedOnly=args.SfixedOnly)
    model.eval()
    model.cuda()
    if args.legacy_mode:
        # Restore some legacy settings for better reproducibility
        model.restore_research_code_artifacts()
    logger.info(f"LVVCP restored Model")

    #Perform inference
    with torch.no_grad():
        last_seq = [None] # some impossilbe name
        seq_frame = 0
        fn_refs_prev = None
        device = "cuda"
        dtype = torch.float32
        if args.verify:
            dtype = torch.float64
        if dtype == torch.float64:
            model = model.to(dtype=dtype)
        logger.info(f"Model running with {model.get_S()} iteration stages")

        t_seq_start = time.time()
        for i_file, file_names in enumerate(file_list):
            #0 - extract names and check if it is a new sequence:
            # Parse filenames to get sequence etc.
            fn_refs, fn_curr = filnamestr_to_refs(file_names)
            c_seq = os.path.split(fn_curr)[0]
            c_file_str = os.path.splitext(os.path.split(fn_curr)[-1])[0]
            logger.info(f"Evaluation: Sequence: '{c_seq}', Frame:{seq_frame}")

            # Load Global reference image(s) from file
            logger.info(f"Next Frame: File Loading: seq:{c_seq}, ref:{fn_refs}, curr:{fn_curr}")
            if fn_refs_prev != fn_refs:
                padder = None # Reset padder object
                global_refs = {
                    'ic':  [None,]*len(fn_refs),   # List of Global refernces - color Lab [N3HW]
                    'ig3': [None,]*len(fn_refs),   # List of Global refernces - gray [N3HW]
                    'im1': [None,]*len(fn_refs),   # [Optional] List of Global refernces - Alpha masks [N3HW]
                }
                # Do the actual image loading + padding
                for i_ref, fn_ref_prev in enumerate(fn_refs):
                    global_refs['ic'][i_ref], global_refs['ig3'][i_ref], global_refs['im1'][i_ref], padder  = \
                        load_img(join( args.dataset_path ,fn_ref_prev), device, dtype, padder=padder)
                file_saver.set_padder(padder)  # pass over padding object to reverse padding on file writing
                fn_refs_prev = fn_refs
                logger.info(f"all refs loaded successfully")
            if not args.allow_alpha_refs:
                global_refs['im1'] = [None for i in enumerate(global_refs['im1'])] # set to None if not alreay None

            # Load new current image to be colorized (can be gray or with gt color)
            # The gt color version is only used for visualization purposes
            i_curr_col_gt, i_curr_gray3, _, _= load_img(join( args.dataset_path ,fn_curr), device, dtype, padder)
            i_curr_luminance = i_curr_col_gt[:,0:1] # extract luminance
            if args.erase_color:
                logger.info(f"ERASING COLOR IN GT, comparison images will only show gray")
                i_curr_col_gt[:,1:3] *= 0
                i_curr_col_gt[:,1:3] += 0.5


            # Check if this is still the same sequence as the previous frame
            if last_seq != c_seq:
                # New sequence detected (global reference frame folder name changed changed)
                logger.info(f"New Sequence Start detected: '{c_seq}'")
                dt = time.time()-t_seq_start
                logger.info(f"Last Sequence '{last_seq}' took {dt}ms to process (incl. file saving) and contained {seq_frame+1} frames {dt/(seq_frame+1)}")
                last_seq = c_seq
                seq_frame = 0
                data_prev = None
                t_seq_start = time.time()
                # New Sequnce: => Use Global Reference as init for current
                i_prev_col   = global_refs['ic'][0].clone()
                i_prev_gray3 = global_refs['ig3'][0].clone()
                # before starting a new sequence - wait until old files of the sequence are saved
                while file_saver.is_busy():
                    logger.info("File saving of current sequence not yet done - waiting another 500ms")
                    time.sleep(0.5)
                file_saver.save_anim()  # Sequence Done, Write video animation
            else:
                # Still old sequence (same reference)
                seq_frame += 1
                # New current Frame:  shift old results to past
                i_prev_col   = last_col_frame_val     # Propagate Refined Color image
                i_prev_gray3 = last_gray_frame_val    # Keep old gray value


            # Defining Data Structure for Color propagation algorithm
            data = {
                'i0c':  i_prev_col,                  # previous coloured image  [N3HW] (Lab)
                'i0g3': i_prev_gray3,                # previous image gray channel  [N3HW] (3xduplicated)

                'i1L':  i_curr_luminance,            # current image luminace channel [N1HW]
                'i1g3': i_curr_gray3,                # current image gray channel (3x) [N3HW] (3xduplicated)
            }

            # Run the actual LVVCP Model on the video image
            x_colorized, res = model.forward( data, data_prev, global_refs, get_res=True)

            #Backup data for next iteration:
            data_prev = data # Make a backup of data
            last_gray_frame_val = i_curr_gray3.clone()
            last_col_frame_val  = x_colorized.clone() # make a backup for propagation

            logger.info(f"Evaluation: Saving Files -- Before")
            out_file_str = f"{seq_frame+1:03}"
            out_file_str = c_file_str # e.g. for blackswan/00000.png this would be 00000
            file_saver( i1c_gt  = data['i0c'],
                            i2c_gt  = i_curr_col_gt,
                            i2c_est = x_colorized,
                            m10_est = res['matched10'][-1],
                            i2c_est_init  = res['i1c'][0],
                            init=seq_frame==0,
                            seq_name= args.seq_path_prefix + c_seq  if args.seq_path_prefix else c_seq,
                            out_file_str=out_file_str)

            data_prev = get_data_for_next_run(data)
            del data

            logger.info(f"Evaluation: Saving Files -- After")

        logger.info(f"Evaluation: finished sequence {c_seq}")

        dt = time.time()-t_seq_start
        logger.info(f"Last Sequence '{last_seq}' took {dt:.2f} s to process (incl. loading & saving) and contained {seq_frame+1} frames, => {dt/(seq_frame+1):.2f} s/frame")

        del res # free memory

    #END ValLoop
    logger.info(f"Evaluation: Saving sequence videos{c_seq}")
    file_saver.save_anim()

    while file_saver.is_busy():
        logger.info("File saving no yet done - waiting 500ms")
        time.sleep(0.5)
    file_saver.shutdown()
    
    logger.info("Evaluation finished")

    logger.info(f"Results can be found under:\n\n  {save_path}\n")



if __name__ == '__main__':
    pass
    parser = argparse.ArgumentParser(
        description=
        """ Simple Demo application for the LVVCP - Learned Variational Video Color Propagation.

            Requires:
                - pre-trained model.pth file + json-config (model.json)
                - validation file list txt file
                - a dataset (a sequence of consecutive gray frames with a colorized reference frame)

            Example:
                python demo_LVVCP.py \
                    --load_path=${WORKING_DIR}/data/DAVIS2017_scootergray/res_mm1_orig_imgs/model/mm1__ep400_3Cands12ItersModel.ckpt \
                    --dataset_path=${WORKING_DIR}/data/ \
                    --list_val=${WORKING_DIR}/data/DAVIS2017_scootergray/scootergray.txt \
                    --out_path=${WORKING_DIR}/out_val/ \
                    --minimal_output=1 \
                    --filesaver_type=MP \

        """
    )   
    parser.add_argument('--load_path',type=str, default='', help='load_path for model')
    parser.add_argument('--out_path',type=str, default='out_val/', help='default outpath prefix for model')
    parser.add_argument('--out_postfix',type=str, default='', help='additional postifx for the save path')
    parser.add_argument('--load_type',type=str, default='', help=f"load type: 'std_tdv', 'vidcol'")
    parser.add_argument('--dataset_path',type=str, default='', help='dataset path')

    parser.add_argument('--filesaver_type',type=str,default='MP', help='MP/std/none, Which Visualizer to use, Multiprocessing or Standard')
    parser.add_argument('--minimal_output',type=int,default=1, help='0/1 if 1 then only a single output is generated')
    parser.add_argument('--verbose',type=int,default=1, help='Show Verbose output during training')
    parser.add_argument('--cand_count',type=int,default=-1, help='-1 = use value from config, else use the amount specified here')
    parser.add_argument('--cfg_name',type=str,
                         default='cfg_VidCol_Lab_base',
                          help='which base config to use')
    parser.add_argument('--cfg_strict_reload'   ,type=int, default=0, help='Only reload cfg, do not merge with new configs => no automatic forward ccompatibility, but safer')
    parser.add_argument('--SfixedOnly',type=float, default=-1, help='Set output to a fixed number of steps')
    parser.add_argument('--erase_color',type=float, default=0, help='If a color sequence is fed, the ground-truth color is shown as comparison, use this option to only show gray instead')
    parser.add_argument('--list_val'   ,type=str, default="", help='Provide an additional val list to evaluate over')
    parser.add_argument('--allow_alpha_refs'   ,type=str, default="", help='Allow usage of Alpha masks in reference (ref must a FULL grayscale image, with and additional alpha mask (1=keep) allowing to scale importance of reference')
    parser.add_argument('--skip_automatic_prefix',type=int, default=0, help='Do not add date etc.')
    parser.add_argument('--out_all_last_seq',type=str, default=0, help='Outputs all data for last sequence')
    parser.add_argument('--seq_path_prefix',type=str, default='', help='add additional string before sequnce path')


    parser.add_argument('--legacy_mode' ,type=int, default=1, help='Run in legacy mode for better reproducibility')

    parser.add_argument('--verify' ,type=int, default=0, help='Check against saved values')
    parser.add_argument('--verify_save' ,type=int, default=0, help='Model verificatoin - generate new outputs')

    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        logging.getLogger("main-logger").error(f"Exception occured: {e}")
        if not sys.platform.startswith("win"):
            s = os.statvfs("/dev/shm")
            logging.getLogger("main-logger").error(f"Free Memory on /dev/shm: {s.f_bfree*s.f_bsize/1024/1024/1024:.3} GB")
        raise