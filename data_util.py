import numpy as np
import cv2
import skvideo.io
import pandas as pd
import os
from shutil import copy
import cv2
import matplotlib.pyplot as plt

def video2matrix(path):#returns shape: frames x height x width x color_dims
    videodata = skvideo.io.vread(path)
    return np.array(videodata)

def action2hot(action_list, aux_dict):
    target_matrix = np.zeros((len(action_list), 101))
    for i in range(len(action_list)):
        target_matrix[i, aux_dict[action_list[i]]] = 1
    return target_matrix

def action2label(action_list, aux_dict):
    target_list = []
    for i in range(len(action_list)):
        target_list.append( aux_dict[action_list[i]])
    return target_list

def getauxdict(original_UCF_path):
    aux_dict = {}
    for action_name in os.listdir(original_UCF_path):
        if(action_name not in aux_dict):
            aux_dict[action_name] = len(aux_dict)
    return aux_dict