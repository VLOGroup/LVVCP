from imageio import imread, imwrite
import numpy as np
from glob import glob
import os, sys
from os import mkdir
import os.path as p
from skimage.transform import resize
import multiprocessing
import shutil


_lib = os.path.join( os.path.dirname(os.path.abspath(__file__)),"../")
if _lib not in sys.path:
    print(_lib)
    sys.path.insert(0, _lib) 
from common_utils.color_conversion_pytorch import rgb2gray_viaLab_NHWC_uint8_np


def load_list(fn, subpath=""):
    with open(fn) as fp:
        lines = [p.join(subpath,l.strip()) for l in fp.readlines()]
        lines.sort()
    return lines

class Downsampler():
    def __init__(self, in_path, out_path, split, H_out, max_file_cnt ):
        self.in_path = in_path
        self.out_path = out_path
        self.split = split
        self.max_file_cnt = max_file_cnt
        self.H_out = H_out

    def __call__(self, dir_name):
        out_path = self.out_path
        in_path = self.in_path
        split = self.split
        H_out = self.H_out
        dir_in_path  = p.join(in_path, dir_name  )
        dir_out_path = p.join(out_path, split,  dir_name  )
        if not p.isdir(dir_out_path):
            os.makedirs(dir_out_path)

        files = glob( dir_in_path + "/*.jpg")
        files.sort()
        files = files[0:self.max_file_cnt]
        sizes = []

        print(f"processing {dir_name} found {len(files)} files")
        for idf, fn_in in enumerate(files):
            fn_path, fn = p.split(fn_in)
            fn, ext = p.splitext(fn)
            fn_out = p.join(dir_out_path, f"{fn}.png")

            if not os.path.isfile(fn_in):
                continue

            img = imread(fn_in)/255.0

            H,W,C = img.shape
            scale = H/H_out
            W_out = np.ceil(W / scale).astype(np.int)
            
            img_down_float = resize(img, (H_out,W_out,3), order=1, anti_aliasing=True)
            img_down_uint8 = np.clip(img_down_float      * 255, 0 ,255).astype(np.uint8)
                
            imwrite(fn_out, img_down_uint8)

            im2gray_methods = [

                {'fn': '_1SimGray_SKimageViaLab_uint8_np', 'im2gray': lambda ifloat,iuint8:  rgb2gray_viaLab_NHWC_uint8_np(iuint8)[...,0] },
            ] 
            for idgray, im2gray_method in enumerate(im2gray_methods):
                img_down_gray_uint8 = im2gray_method['im2gray'](img_down_float, img_down_uint8)

                dir_out_path_gray =  p.join(out_path, split+ im2gray_method['fn'],  dir_name )
                fn_out_gray = p.join(dir_out_path_gray, f"{fn}.png")
                if idf == 0:
                    if not p.isdir(dir_out_path_gray):
                        os.makedirs(dir_out_path_gray)
                    imwrite(fn_out_gray, img_down_uint8)  # First Frame is colour             
                    imwrite(fn_out_gray.replace(".png","_gray.png"), img_down_gray_uint8)
                else:
                    imwrite(fn_out_gray, img_down_gray_uint8)

            sizes.append({dir_name: [fn, (H,W), (H_out,W_out)]})
        
        return sizes


if __name__ == "__main__":

    base_path = "/mnt/Data4TB/Datasets/davis/"

    IMG_CNT_VAL=25
    IMG_CNT_TRAIN=25
    in_path = f"{base_path}/DAVIS-2017-trainval-Full-Resolution/DAVIS/JPEGImages/Full-Resolution/"
    out_path = f"{base_path}/DAVIS-2017-trainval-480p_downsampled_png_v3/"
    val_lst = f"{base_path}/vpn_main_val.txt"
    train_lst = f"{base_path}/vpn_main_train.txt"
    fn_lists = {
        'val':val_lst,
        'train':train_lst
    }

    # IMG_CNT_VAL=50
    # in_path = f"{base_path}/DAVIS-2019-Unsupervised-test-dev-Full-Resolution/DAVIS/JPEGImages/Full-Resolution/"
    # out_path = f"{base_path}/DAVIS-2019-testdev-480p_downsampled_png/"
    # test_lst = f"{base_path}/DAVIS-2019-Unsupervised-test-dev-Full-Resolution/DAVIS/JPEGImages/Full-Resolution/sequence_names.txt"
    # train_lst = f"{base_path}/DAVIS-2019-Unsupervised-test-dev-Full-Resolution/DAVIS/JPEGImages/Full-Resolution/sequence_names.txt"
    # fn_lists = {
    #     'test':test_lst,
    # }

    # IMG_CNT_VAL=50
    # in_path = f"{base_path}/DAVIS-2017-test-dev-Full-Resolution/DAVIS/JPEGImages/Full-Resolution/"
    # out_path = f"{base_path}/DAVIS-2017-testdev-480p_downsampled_png/"
    # test_lst = f"{base_path}/DAVIS-2017-test-dev-Full-Resolution/DAVIS/JPEGImages/Full-Resolution/sequences.txt"
    # fn_lists = {
    #     'test':test_lst,
    # }


    fn_val_lst   = load_list(val_lst )
    fn_train_lst = load_list(train_lst)

    H_out = 480

    lists = {key: load_list(lst) for key,lst in fn_lists.items()}

    max_file_cnt = {
        'val':IMG_CNT_VAL, 
        'train':IMG_CNT_TRAIN,
        'test':50,
    }

    pool = multiprocessing.Pool(processes=8)

    for split, dir_lst in lists.items():
        func = Downsampler(in_path, out_path, split, H_out, max_file_cnt=max_file_cnt[split])
        res = []
        for x in pool.imap(func, dir_lst):
            res.append(x)

            print(x)
        with open(out_path + f"/stats_downsample_{split}.txt","w") as fp:
            fp.write(str(res))
    print("done")