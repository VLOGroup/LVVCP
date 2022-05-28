# Examplary Data, models and results for comparison

The DAVIS2017_scootergray folder contains exemplary data from the DAVIS 2017 dataset from, https://davischallenge.org, which was originally released under [CC BY-NC](http://creativecommons.org/licenses/by-nc/4.0). license.
Please follow the original data authors reqest and cite their papers (see website) if you use their data.


For best quality the 480x854 images (orignally JPG) have been re-sampled from their High-resolution sources using standard tools, anti aliasing filters and saves as lossless PNG format, see [../common_utils/prepare_DavisDataset.py](../common_utils/prepare_DavisDataset.py).
The gray-scale images have been created by using skimages lab2rgb and rgb2lab functions and setting the ab channels to 0, for details see [../common_utils/color_conversion_pytorch.py](../common_utils/color_conversion_pytorch.py).


We provide a model for the best setup of our paper, i.e. using all color proposals and additional multi-model training [data/DAVIS2017_scootergray/res_mm1_ep400/model/mm1__ep400_3Cands12ItersModel.ckpt](data/DAVIS2017_scootergray/res_mm1_ep400/model/mm1__ep400_3Cands12ItersModel.ckpt). Additionaly we provide a fast CUDA sampling operator, which can be used by loading the model from [data/DAVIS2017_scootergray/res_mm1_ep400_CUDA_Op_sample_cv/model_NHWC_bilin/mm1__ep400_3Cands12ItersModel_NHWC.ckpt](data/DAVIS2017_scootergray/res_mm1_ep400_CUDA_Op_sample_cv/model_NHWC_bilin/mm1__ep400_3Cands12ItersModel_NHWC.ckpt)
Using this sample operator provides similar results with faster inference and less GPU memory usage.

