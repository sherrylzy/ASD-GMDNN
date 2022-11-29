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
from torch.utils.data import DataLoader
from torchsummary import summary
########################################################################
# original lib
########################################################################
from Preprocess import common as com
#from ASD.Module.trainprocess import ASD_train_process
from utils.utils import mkdir_p, save_checkpoint, MyOrderedDict
from Module.AEGMM import AEGMM
from Train.aegmm_train import TrainerAEGMM


########################################################################


########################################################################
# load parameter.yaml
########################################################################
yam = "/home/zhaoyi/Project/ASD-AEGMM/ASD/ASDGMM_500.yaml"
param = com.yaml_load(yam)

########################################################################
# load device
########################################################################
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

########################################################################


########################################################################
# visualizer
########################################################################
class visualizer(object):
    def __init__(self):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.fig = self.plt.figure(figsize=(7, 5))
        self.plt.subplots_adjust(wspace=0.3, hspace=0.3)

    def loss_plot(self, loss, val_loss):
        """
        Plot loss curve.

        loss : list [ float ]
            training loss time series.
        val_loss : list [ float ]
            validation loss time series.

        return   : None
        """
        ax = self.fig.add_subplot(1, 1, 1)
        ax.cla()
        ax.plot(loss)
        ax.plot(val_loss)
        ax.set_title("Model loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(["Train", "Validation"], loc="upper right")

    def save_figure(self, name):
        """
        Save figure.

        name : str
            save png file path.

        return : None
        """
        self.plt.savefig(name)


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


########################################################################


########################################################################
# main AE+GMM_train.py
########################################################################
if __name__ == "__main__":
    # check mode
    # "development": mode == True
    # "evaluation": mode == False
    mode = com.command_line_chk()
    if mode is None:
        sys.exit(-1)

    # make output directory
    os.makedirs(param["model_directory"], exist_ok=True)

    # initialize the visualizer
    visualizer = visualizer()

    # load base_directory list

    dirs = com.select_dirs(param=param, mode=mode)

    # loop of the base directory
    for idx, target_dir in enumerate(dirs):
        print("\n===========================")
        print("[{idx}/{total}] {target_dir}".format(target_dir=target_dir, idx=idx + 1, total=len(dirs)))

        # set path
        machine_type = os.path.split(target_dir)[1]
        model_file_path = "{model}/model_{machine_type}_{epoch}_aegmm.pth".format(model=param["aegmm"]["save_path"],
                                                                                  machine_type=machine_type,
                                                                                  epoch=param["aegmm"]["num_epochs"])

        if os.path.exists(model_file_path):
            com.logger.info("model exists")
            continue

        history_img = "{img}/history_{machine_type}_{epoch}.png".format(img=param["aegmm"]["img_dir"],
                                                                  machine_type=machine_type,
                                                                    epoch = param["aegmm"]["num_epochs"])
        # pickle file for storing anomaly score distribution
        score_distr_file_path = "{model}/score_distr_{machine_type}.pkl".format(model=param["aegmm"]["save_path"],
                                                                                machine_type=machine_type)

        # generate dataset
        print("============== DATASET_GENERATOR ==============")

        # get file list for all sections
        # all values of y_true are zero in training
        files, y_true = com.file_list_generator(target_dir=target_dir,
                                                section_name="*",
                                                dir_name="train",
                                                mode=mode)
        # test
        #files = files[0:400]
        #y_true = y_true[0:400]

        # 1. make feature for predict
        #files=files.to(device=device_ids[0])
        #y_true = y_true.to(device=device_ids[0])
        feats = file_list_to_astdata(files,
                                    msg="generate train_dataset",
                                    n_mels=param["feature-ast"]["n_mels"],
                                    n_frames=param["feature-ast"]["n_frames"],
                                    n_hop_frames=param["feature-ast"]["n_hop_frames"],
                                    n_fft=param["feature-ast"]["n_fft"])

        #number of vectors for each wave file
        #n_vectors_ea_file = int(data.shape[0] / len(files))
        input_tdim = int(feats.shape[1])
        feats_data = torch.from_numpy(feats)
        feats_data = feats_data.unsqueeze(1)
        #feats_data = feats_data.transpose(0, 1)
        train_data =DataLoader(feats_data, batch_size=param["aegmm"]["batch_size"],
                              shuffle=True, num_workers=0)

        # 2. train AE
        #model = AEGMM().to(device)
        #summary(model, (2, 1, 1024, 128).to(device))

        JAEGmm = TrainerAEGMM(param["aegmm"], train_data, device, history_img)

        JAEGmm.train()
        
        torch.save(JAEGmm.model.state_dict(),
                   os.path.join(model_file_path))
        torch.cuda.empty_cache()

        #criterion_MSE = nn.MSELoss()
        #criterion_COS = nn.CosineEmbeddingLoss()
        #Tar = torch.tensor([1, -1])
        '''
        if torch.cuda.is_available():
            JAEGmm.cuda()
        for epoch in range(epoches):
            if epoch in [epoches * 0.25, epoches * 0.5]:
                for param_group in optimizier.param_groups:
                    param_group['lr'] *= 0.1
                # train
                #output_CN, hidden = JAEGmm(train_data) # hidden: encoder output embedding
                #rec_euclidean = criterion_MSE(output_CN, train_data)
                train_data_feature = train_data.view(batch_size, 1024*128)
                test = train_data_feature.view(batch_size, 1024, 128)
                print(train_data-test)
                #output_AEGMM_feature = output_CN.view(batch_size, 1024*128)
                #rec_cosine = criterion_COS(output_AEGMM_feature, train_data_feature, Tar)

                # backward
                optimizier.zero_grad()
                rec_euclidean.backward()
                optimizier.step()
                '''





















