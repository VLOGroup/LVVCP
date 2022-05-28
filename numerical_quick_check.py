# This File is meant for a quick check to see if the installed setup gives roughly the expected numerical results.
#

from imageio import imread
from skimage.color import rgb2lab, lab2rgb    
from common_utils.psnr_utils import compute_PSNRab_th, compute_PSNRab_np, get_ab_max_full_swing
import pandas as pd
from os.path import join, split
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

def get_paths_from_args():
    parser = ArgumentParser()
    parser.add_argument('--sample_cv', type=int, 
                        default=0,
                        # default=1,
                        help='Specify wether the CUDA sample_cv operator was used to compute the results')
    parser.add_argument('--res_path', type=str,
                        # default='',
                        default = 'out_val/vis_2022_0711_2329_cfg_MultiModelSP_Glob_GlobLoc__Ftrain9bs4mm1__ep400_3Cands12ItersModel_/',
                        # default = 'out_val/vis_2022_0711_2330_cfg_MultiModelSP_Glob_GlobLoc__Ftrain9bs4mm1__ep400_3Cands12ItersModel_NHWC_/',
                        help='The path to the results from demo_scripts/demo_LVVCP_scootergray.sh or demo_scripts/demo_LVVCP_scootergray_cuda.sh  \n'
                             'Typically something like out_val/vis_2022_0711_2330_cfg_MultiModelSP_Glob_GlobLoc__Ftrain9bs4mm1__ep400_3Cands12ItersModel_NHWC_/')
    parser.add_argument('--res_base_path', type=str,
                        default='data_and_model_samples/DAVIS2017_scootergray/',
                        help='The base path to the results for comparison')
    parser.add_argument('--res_postfix_path', type=str,
                        default='DAVIS2017_scootergray/gray/',
                        help='The default postfix path to the results for comparison')
    args = parser.parse_args()


    sub_path_std = 'res_mm1_ep400/imgs/'
    sub_path_cuda_sample_cv = 'res_mm1_ep400_CUDA_Op_sample_cv/imgs/'

    paths = {}
    if args.sample_cv:
        paths['imgs_provided'] = join(args.res_base_path, sub_path_cuda_sample_cv)
    else:
        paths['imgs_provided'] = join(args.res_base_path, sub_path_std)
        
    paths['gt'] = join(args.res_base_path,'gt/')
    paths['csv'] = join(paths['imgs_provided'], '../res.csv')
    paths['imgs_generated'] =  join(args.res_path, args.res_postfix_path)

    return paths


if __name__ == '__main__':
    paths = get_paths_from_args()

    # read expected PSNR_ab results from csv (to check if skimage etc. ran ok)
    df_res = pd.read_csv(paths['csv'])


    print(f"   |     CSV    |   img old  |  new data  |  delta to CSV  | delta to old  |")      # PSNR from new data 
    print(f"   |   PSNR_ab  |   PSNR_ab  |   PSNR_ab  |                | provided img  | ")      # PSNR from new data 
    print(f"---+------------+------------+------------+----------------+---------------+")
    for i in range(1,len(df_res)+1):
        fn_expted = join(paths['imgs_provided'],f"{i:05}.png")
        fn_gt  = join(paths['gt'],f"{i:05}.png")
        fn_res = join(paths['imgs_generated'],f"{i:05}.png")
        img_expted = imread(fn_expted) / 255.0
        img_gt     = imread(fn_gt) / 255.0
        img_res    = imread(fn_res) / 255.0
        img_expted_lab = rgb2lab(img_expted)
        img_gt_lab     = rgb2lab(img_gt)
        img_res_lab    = rgb2lab(img_res)

        PSNR_CSV = df_res[df_res['frame'] == float(i)]['ab_PSNR_full'].iloc[0]             # PSNR from CSV file
        PSNR_prevFile = compute_PSNRab_np( img_expted_lab[...,1:3], img_gt_lab[...,1:3])   # PSNR from provided image
        PSNR_res    = compute_PSNRab_np( img_res_lab[...,1:3], img_gt_lab[...,1:3])        # PSNR from new data 
        print(f"{i:2} | {PSNR_CSV:0.5f} | {PSNR_res:0.5f} | {PSNR_res:0.5f} |      {PSNR_res-PSNR_CSV:+0.3f}    |     {PSNR_res-PSNR_prevFile:+0.3f}    |")

    print("done")