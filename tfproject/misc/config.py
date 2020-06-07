from yacs.config import CfgNode as CN


# Initial configurations
_C = CN()

_C.ENVIRONMENT = CN()
_C.ENVIRONMENT.NAME = "tfproject" 
_C.ENVIRONMENT.TRAIL_NUMBER = 0 #operating system specific 

_C.OPTIMIZER = CN() #training configurations
_C.OPTIMIZER.LR = 0.0001 #learning rate


def get_cfg_defaults():
    return _C.clone()

def print_h():

    print("hello")
