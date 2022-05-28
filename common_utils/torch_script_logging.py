import torch
import logging
import os

from typing import Type, TypeVar, Generic

class TorchScriptLogger(torch.nn.Module):
    def __init__(self, name):
        """
        A wrapper class that allows to use logging in std. and torchscript code
        """
        super(TorchScriptLogger, self).__init__()
        self.use_log = True
        self.use_print = False
        self.torchscript_export = False
        if 'TORCHSCRIPT_EXPORT_WITH_LOGS' in  os.environ and  os.environ['TORCHSCRIPT_EXPORT_WITH_LOGS']:
            self.use_log = False
            self.use_print = True
            self.torchscript_export = True
        elif 'TORCHSCRIPT_EXPORT_WO_LOGS' in  os.environ and  os.environ['TORCHSCRIPT_EXPORT_WO_LOGS']:
            self.use_log = False
            self.use_print = False
            self.torchscript_export = True
        else:
            self.logger = logging.getLogger(name)

        self._levels = {'CRITICAL':50,
                        'ERROR':40,
                        'WARNING':30,
                        'INFO':20,
                        'DEBUG':10,
                        'NOTSET':0}


    def _print(self, msg:str=''):
        print(msg)

    
    def info(self, msg:str=''):
        if self.use_log:
            self._log(msg,'INFO') 
        elif self.use_print:
            self._print(msg)

    def warning(self, msg:str=''):
        if self.use_log:
            self._log(msg,'WARNING')
        elif self.use_print:
            self._print(msg)


    def _print(self, msg:str=''):
        print(msg)

    @torch.jit.unused
    def _log(self, msg:str='', level_str:str='INFO'):
        """ This function is not available in TorchScript mode => Set TORCHSCRIPT_EXPORT_WITH_LOGS or TORCHSCRIPT_EXPORT_WO_LOGS environment variables"""
        self.logger.log(self._levels[level_str], msg)
        pass


_global_logger = TorchScriptLogger('main-logger')
TORCHSCRIPT_EXPORT =  _global_logger.torchscript_export

if _global_logger.use_log:
    print("TorchScriptLogger - using real logger")
    def log_info(msg:str):
        _global_logger.info(msg)

    def log_warning(msg:str):
        _global_logger.warning(msg)
elif _global_logger.use_print:
    print("TorchScriptLogger - using print function, allowing some kind of loggin in exported torch script")
    def log_info(msg:str):
        print(msg)

    def log_warning(msg:str):
        print(msg)
else:
    print("TorchScriptLogger - deactivating logging compeletely")
    def log_info(msg:str):
        pass

    def log_warning(msg:str):
        pass





if __name__ == '__main__':
    class DemoModel(torch.nn.Module):
        def __init__(self):
            super(DemoModel, self).__init__()
            self.logger =  TorchScriptLogger('main-logger')

        def forward(self, x:int):
            self.logger.info(f"asdf {x}")

    logger = logging.getLogger("main-logger")
    logger.setLevel(logging.DEBUG)
    console_handle = logging.StreamHandler()
    console_handle.setLevel(logging.DEBUG)
    # formatter = logging.Formatter("%(name)-20s - %(levelname)-8s - %(message)s")
    # console_handle.setFormatter(formatter)
    logger.addHandler(console_handle)

    model = DemoModel()
    model(1)

    os.environ['TORCHSCRIPT_EXPORT_WITH_LOGS']='1'
    # os.environ['TORCHSCRIPT_EXPORT_WO_LOGS']='1'
    

    torchscript_logger = torch.jit.script(model)
    torchscript_logger.save('./tmp.pt')

    torchscript_logger = torch.jit.load('./tmp.pt')

    torchscript_logger(10)


