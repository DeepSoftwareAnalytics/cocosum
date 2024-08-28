import torch
import os
from util.LoggerUtil import *

device = None
USE_GPU = False


def set_device(gpu_id):
    global device
    global USE_GPU
    if gpu_id == -1:
        device = torch.device("cpu")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        if torch.cuda.is_available():
            device = torch.device("cuda:" + str(gpu_id))
            USE_GPU = True
        else:
            info_logger("Cannot find GPU id %s! Use CPU." % gpu_id)
            device = torch.device("cpu")

    info_logger("[Setting] device: %s" % device)


def move_to_device(data):
    global device
    global USE_GPU

    if USE_GPU:
        return data.cuda(device, non_blocking=True)
    else:
        return data

def move_pyg_to_device(data):
    global device
    global USE_GPU
    
    if USE_GPU:
        return data.to(device)
    else:
        return data

def move_model_to_device(model):
    global device
    global USE_GPU

    if USE_GPU:
        model.to(device)
