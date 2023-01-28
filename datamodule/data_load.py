########################################################################
# import default libraries
########################################################################
import os
import sys
import gc
########################################################################


########################################################################
# import additional libraries
########################################################################
import numpy as np
from sympy import expand
from collections import OrderedDict
import collections
import scipy.stats
# from import
from tqdm import tqdm
try:
    from sklearn.externals import joblib
except:
    import joblib
########################################################################
# torch lib
########################################################################
import torch.optim as optim
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
########################################################################
# original lib
########################################################################
from ASD.Preprocess import common as com
from ASD.Module.ast_models import ASTModel, PatchEmbed
#from ASD.Module.trainprocess import ASD_train_process
from ASD.utils.utils import mkdir_p
from ASD.utils.utils import save_checkpoint
from ASD.utils.utils import MyOrderedDict
from ASD.Module.AEGMM import AEGMM


########################################################################


########################################################################
# load parameter.yaml
########################################################################
param = com.yaml_load()
########################################################################
# visualizer
########################################################################

########################################################################


########################################################################
# get data from the list for file paths
########################################################################
def file_list_to_astdata(file_list,
                         msg="calc...",
                         n_mels=128,
                         n_frames=1,
                         n_hop_frames=1,
                         n_fft=1024):
    """
    convert the file_list to a vector array.
    file_to_vector_array() is iterated, and the output vector array is concatenated.

    file_list : list [ str ]
        .wav filename list of dataset
    msg : str ( default = "calc..." )
        description for tqdm.
        this parameter will be input into "desc" param at tqdm.

    return : numpy.array( numpy.array( float ) )
        data for training (this function is not used for test.)
        * dataset.shape = (number of feature vectors, dimensions of feature vectors)
    """
    # calculate the number of dimensions
    #dims = n_mels * n_frames
    # iterate AST feature ()
    vectors_temp = []
    data = []
    for idx in tqdm(range(len(file_list)), desc=msg):
        vectors = com.file_to_AST(file_list[idx],
                                  n_mels=n_mels,
                                  n_fft=n_fft)


        if idx == 0:
            data = vectors
        else:
            data  = np.vstack((data, vectors))

    data = data.reshape(len(file_list), n_fft, n_mels)

    return data

########################################################################
# get data from the list for file paths
########################################################################
def file_list_to_data(file_list,
                      msg="calc...",
                      n_mels=64,
                      n_frames=5,
                      n_hop_frames=1,
                      n_fft=1024,
                      hop_length=512,
                      power=2.0):
    """
    convert the file_list to a vector array.
    file_to_vector_array() is iterated, and the output vector array is concatenated.

    file_list : list [ str ]
        .wav filename list of dataset
    msg : str ( default = "calc..." )
        description for tqdm.
        this parameter will be input into "desc" param at tqdm.

    return : numpy.array( numpy.array( float ) )
        data for training (this function is not used for test.)
        * dataset.shape = (number of feature vectors, dimensions of feature vectors)
    """
    # calculate the number of dimensions
    dims = n_mels * n_frames

    # iterate file_to_vector_array()
    for idx in tqdm(range(len(file_list)), desc=msg):

        vectors = com.file_to_vectors(file_list[idx],
                                      n_mels=n_mels,
                                      n_frames=n_frames,
                                      n_fft=n_fft,
                                      hop_length=hop_length,
                                      power=power)
        vectors = vectors[:: n_hop_frames, :]
        if idx == 0:
            data = np.zeros((len(file_list) * vectors.shape[0], dims), float)
        data[vectors.shape[0] * idx: vectors.shape[0] * (idx + 1), :] = vectors

    return data

class dataload:
    def __init__(self, data_dir, mode):
        for idx, target_dir in enumerate(data_dir):
            files, y_true = com.file_list_generator(target_dir=target_dir,
                                                    section_name="*",
                                                    dir_name="train",
                                                    mode=mode)

