import errno
import os
import sys
import torch
import numpy as np
import git
import platform
import logging
import pprint
import json
from ast import literal_eval
from copy import deepcopy

UINT16MAX = 2**16 -1
UINT8MAX = 2**8 -1

class DictAverager(object):
    """Computes the Average over dictionary entries """
    def __init__(self):
        self.data_vals = {}
        self.data_cnts = {}
        self.first_run = True
    def update(self, data_dict, n=1):
        if self.first_run:
            for key,val in data_dict.items():
                self.data_vals[key] = val
                self.data_cnts[key] = n
        else:
            for key,val in data_dict.items():
                self.data_vals[key] += val
                self.data_cnts[key] += n
    @property
    def avg(self):
        """returns a full dictionary with average values """ 
        avgs = {}
        for key,val in self.data_vals.items():
            avgs[key] = val / self.data_cnts[key]
        return avgs

    def get_avg(self, key):
        """returns the average of a single data entry in the dictionary""" 
        assert key in self.data_vals, f"key not found {key}"
        return self.data_vals[key] / self.data_cnts[key]


class bcolors:
    """https://stackoverflow.com/a/287944/5010481"""
    # TODO: Requires Python 3.8 and Final[str] annotation for torchscript export
    HEADER:str = '\033[95m'
    OKBLUE:str = '\033[94m'
    OKGREEN:str = '\033[92m'
    WARNING:str = '\033[93m'
    FAIL:str = '\033[91m'
    ENDC:str = '\033[0m'
    BOLD:str = '\033[1m'
    UNDERLINE:str = '\033[4m'

class CUDA_Timer(object):
    def __init__(self, record=True):

        self._timers=[]
        self._do_record = record
    
    def record_time(self, msg=""):
        if self._do_record:
            self._timers.append( (torch.cuda.Event(enable_timing=True),
                                msg)
                            )
            self._timers[-1][0].record()

    def get_timings(self):
        if self._do_record:
            torch.cuda.synchronize()
            assert len(self._timers) > 1, f"must call add_time at least twice!"

            timing_strs = []
            for idx,t in enumerate(self._timers):
                if idx >= len(self._timers)-1:
                    break
                tprev,msg_prev = t
                tcurr,msg_curr = self._timers[idx+1]
                dt_ms = tprev.elapsed_time(tcurr)
                dt_str = f"{dt_ms:8.2f}ms, {msg_prev} -> {msg_curr}"
                timing_strs.append(dt_str)


            return "\n".join(timing_strs)
        else:
            return "timing omitted - set Timer.record to True"


def deepcopy2cpu(dat, idb=None):
    """ takes a datastructure possibly on GPU and creates a copy on the CPU
        
        by Default all parameters will be copied. 
        If idb (int) is specified, Tensor types will be assumed to be batches and sliced accoringly.
        idb=0 will select the batch with index 0, and keeping a 1 dimension.
    """
    if type(dat) is torch.Tensor:
        if idb is None:
            return dat.detach().cpu()
        else:
            if dat.ndim == 0:
                return dat.detach().cpu()
            else:
                return dat[idb:idb+1].detach().cpu()
    elif isinstance(dat, torch.nn.parameter.Parameter):
        return dat.data.detach().cpu()
    elif type(dat) in [list, tuple]:
        return [deepcopy2cpu(elem, idb) for elem in dat]
    elif isinstance(dat, dict): # a dictionary type
        out_dict = type(dat)()  # make a empty new version of dict, OrderedDict,....
        for key, elem in dat.items():
            out_dict[key] =  deepcopy2cpu(elem, idb)
        return out_dict
    elif type(dat) in [int, float, str, bool]:
        return dat
    else:
        raise ValueError(f"found a not yet supported type {type(dat)} ")


def pt2np(t, squeeze=False):
    """ Converts a pytorch tensor NCHW or CHW to NHWC or HWC
        if squeeze=True   N=1 and C=1 will be removed,
    """
    if t.ndim == 3:
        out = t.permute(1,2,0).detach().cpu().numpy()
        if squeeze and out.shape[2] == 1:
                out = np.squeeze(out,2)
    elif t.ndim == 4:
        out = t.permute(0,2,3,1).detach().cpu().numpy()
        if squeeze and out.shape[3] == 1:
            out = np.squeeze(out,3)
        if squeeze and out.shape[0] == 1:
            out = np.squeeze(out,0)
    return out

def np2uin8img(arr, clamp=True, compress=False):
    if compress:
        arr = arr - arr.min()
        arr = arr / arr.max()
        arr = arr * 255
    arr = np.clip(arr,0,255)
    arr = arr.astype(np.uint8)
    return arr

_UINT16_MAX = 2**16-1
def np2uint16img(arr, clamp=True, compress=False):
    if compress:
        arr = arr - arr.min()
        arr = arr / arr.max()
        arr = arr * _UINT16_MAX
    arr = np.clip(arr,0,_UINT16_MAX)
    arr = arr.astype(np.uint16)
    return arr

def th_to_uint16(tensor: torch.Tensor) -> torch.Tensor:
    """ returns a tensor converted to uint16 (as used for uint16 png file storage) inside a int32 container"""
    return torch.clamp(_UINT16_MAX * tensor,0,_UINT16_MAX).to(dtype=torch.int32)

def uint16_to_th(uint16: torch.Tensor) -> torch.Tensor:
    """takes an uint16 tensor (as used for uint16 png file storage) and converts it back to a float tensor"""
    return uint16.float() / _UINT16_MAX

def get_bytes_count(tensor):
    return tensor.element_size() * tensor.nelement()

def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def check_makedirs(dir_name):
    if not os.path.exists(dir_name):
        try:
            os.makedirs(dir_name)
        except OSError as e:
            # In unlikely case that another process does the same at the same time
            if e.errno != errno.EEXIST:
                raise
        

def log_config(logger, config, filepath, post_fix=""):
    cfg_str = pprint.pformat(config)
    cfg_str = f"CONFIG_STR_BEGIN:\nconfig={cfg_str}\nCONFIG_STR_END"
    logger.info(cfg_str)
    with open(os.path.join(filepath,f"config{post_fix}.json"), "w") as fp:
        json.dump(config, fp)
    with open(os.path.join(filepath,f"config{post_fix}.log"),"w+") as fp:
        fp.write(cfg_str)

def log_dataloader(logger, loader, filepath, post_fix=""):
    log_str = f"#Train Loader:\n{pprint.pformat(loader)}"
    log_str = f"DATALOADER_STR_BEGIN:\n{log_str}\nDATALOADER_STR_END"
    logger.info(log_str)
    with open(os.path.join(filepath,f"info__dataloader{post_fix}.log"),"w+") as fp:
        fp.write(log_str)

def log_info(msg):
    logging.getLogger("main-logger").info(msg)

def log_warning(msg):
    logging.getLogger("main-logger").warning(msg)

# logger
def get_logger(args, filepath=None):
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    if filepath:
        if not os.path.isdir(filepath):
            os.mkdir(filepath)
        fh = logging.FileHandler(f'{filepath:s}/logging.log')
        fh.setFormatter(logging.Formatter(fmt))
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)

    logger.info(args)
    logger.info(f"Running on Node: '{platform.uname().node}' ")
    logger.info(f"Python executable: '{sys.executable}' ")
    logger.info(f"Visible GPUs: '{os.environ.get('CUDA_VISIBLE_DEVICES','')}'")
    logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
    if hasattr(args,'cfg_name'):
        logger.info(f"Base Config: {args.cfg_name}")
    if filepath:
        # SAVE GPUs used
        fn = f"{filepath:s}/info__usded_GPUs_{os.environ.get('CUDA_VISIBLE_DEVICES','')}.log"
        with open(fn, "w") as fp:
            fp.write(f"Visible GPUs: '{os.environ.get('CUDA_VISIBLE_DEVICES','')}'")
        # Save Environment Variables Present
        with open(f"{filepath:s}/info__environment_variables.log", "w") as fp:
            fp.write(f"Environment Variables found:\n{pprint.pformat(os.environ._data)}")
        # Save Calling arguments
        with open(f"{filepath:s}/info__call_arguments.log", "w") as fp:
            fp.write(f"call_args={pprint.pformat(dict(args._get_kwargs()))}")
    if 'SLURM_JOB_ID' in os.environ:
        SLURM_EVNRION =  {k:v for k,v in os.environ.items() if k.startswith("SLURM_")}
        logger.info(f"Running in Slurm, JOBID: '{os.environ['SLURM_JOB_ID']}' ")
        with open(f"{filepath:s}/info__slurm_settings.log", "w") as fp:
            fp.write(f"Slurm settings GPUs (SLURM): \n{pprint.pformat(SLURM_EVNRION)}")
        if 'SLURM_STEP_GPUS' in SLURM_EVNRION:
            fn = f"{filepath:s}/info__usded_GPUs_SLURM_{os.environ.get('SLURM_STEP_GPUS','')}.log"
            with open(fn, "w") as fp:
                fp.write(f"Visible GPUs (SLURM): '{os.environ.get('SLURM_STEP_GPUS','')}'")

    try:
        repo = git.Repo(".")
        if repo.head.is_detached:
            logger.info(
                f"git stats: branch=DETACHED_HEAD_NO_BRANCH, hexsha={repo.head.object.hexsha}, modified files={[item.a_path for item in repo.index.diff(None)]}")
        else:
            logger.info(
                f"git stats: branch={repo.active_branch.name}, hexsha={repo.active_branch.object.hexsha}, modified files={[item.a_path for item in repo.index.diff(None)]}")
    except git.exc.InvalidGitRepositoryError:
        logger.info(f"git stats: did not find a git repository")

    return logger


def psnr(x, x_gt, max_gt = None): 
    if max_gt is None:
        return 20*torch.log10(x_gt.max()/torch.sqrt(torch.mean((x-x_gt) ** 2)))
    else:
        return 20*torch.log10(max_gt/torch.sqrt(torch.mean((x-x_gt) ** 2)))




def load_cfg_log(cfg_log_path):
    """ loads a config from a 'config.log' file"""
    assert os.path.isfile(cfg_log_path), f"could not find {cfg_log_path}"
    with open (cfg_log_path) as fp:
        print("loading config from config.log")
        assert "CONFIG_STR_BEGIN:" in fp.readline() , f"missing file header must start with 'CONFIG_STR_BEGIN:'"
        txt = fp.readlines()
        assert "CONFIG_STR_END" in txt[-1], f"missing file end"
        txt = "".join(txt [0:-1])
        assert txt[0:7] =="config="
        txt = txt[7:-1].replace("\n","")
    
    config = literal_eval(txt)
    return config

def join_dicts(new_style, old_style, cp_exceptions):
    """ copy old_style dict into new one, overwriting values if not in exception""" 
    for key in old_style:
        try:
            if key in cp_exceptions:
                continue # keep new settings here
            if key not in new_style:
                if type(old_style[key]) == dict:
                    if key not in new_style:
                        new_style[key] = {}
                    join_dicts(new_style[key], old_style[key],cp_exceptions)
                else:
                    new_style[key] = old_style[key]
                    print(f"added: {key}: {new_style[key]}=>{old_style[key]}")
            elif  new_style[key] != old_style[key]:
                if type(old_style[key]) == dict:
                    if key not in new_style:
                        new_style[key] = {}
                    join_dicts(new_style[key], old_style[key],cp_exceptions)
                else:
                    new_style[key] = old_style[key]
                    print(f"added: {key}: {new_style[key]}=>{old_style[key]}")
        except:
            print(f"-----\nEXCEPTION: join_dicts failed: {key}\n new_dict:\n{new_style}\n old_dict:\n{old_style}\n---------\n")
            raise

def get_min_value_lastdim(vals, condition, get_best_cond_val: bool=False):
    """ Extracts values based on a condition.
        This operation is broadcastable,
          i.e. the condition can have 1 dimensions whereas the vals can have larger corresponding dimensions
    """
    assert vals.ndim == condition.ndim
    for v_s, c_s in zip(vals.shape, condition.shape):
        if not ((v_s == c_s) or (c_s ==1)):
            raise ValueError(f"dimensions must be identical or broadcastable val.shape={vals.shape}, condition={condition.shape}")
    sizes = torch.prod(torch.tensor(vals.shape[0:-1]))
    if condition.shape != vals.shape:
        condition = condition + torch.zeros_like(vals) # broadcast condition
    if get_best_cond_val:
        cond_vals_best, id_min = torch.min(condition,dim=-1)
    else:
        id_min = torch.argmin(condition,dim=-1)
        cond_vals_best = None
    id_sz_flat = torch.arange(sizes, device=vals.device)
    val_best = vals.reshape(sizes, vals.shape[-1])[id_sz_flat, id_min.flatten()]
    val_best = val_best.reshape(vals.shape[0:-1])

    return val_best, cond_vals_best



def get_min_cand_v1(col_cands, delta_guide):
    """Selects the best candidate using a guide, looks at last dimension """
    val_min_guide, idmin_guide = delta_guide.min(-1)
    B,C,H,W,cnt = col_cands.shape
    idmin_guide = idmin_guide.repeat(1,C,1,1).reshape(-1)
    ids_flat = torch.arange(0,B*C*H*W,device=col_cands.device)
    col_cands_flat = col_cands.reshape(B*C*H*W,cnt)
    col_cand = col_cands_flat[ids_flat, idmin_guide]
    col_cand = col_cand.reshape(B,C,H,W)
    return col_cand, val_min_guide


def get_max_cand_v1(col_cands, delta_guide):
    """Selects the best candidate using a guide, looks at last dimension """
    val_max_guide, idmax_guide = delta_guide.max(-1)
    B,C,H,W,cnt = col_cands.shape
    idmax_guide = idmax_guide.repeat(1,C,1,1).reshape(-1)
    ids_flat = torch.arange(0,B*C*H*W,device=col_cands.device)
    col_cands_flat = col_cands.reshape(B*C*H*W,cnt)
    col_cand = col_cands_flat[ids_flat, idmax_guide]
    col_cand = col_cand.reshape(B,C,H,W)
    return col_cand, val_max_guide
