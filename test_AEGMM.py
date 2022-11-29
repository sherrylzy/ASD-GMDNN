########################################################################
# import default libraries
########################################################################
import os
import csv
import sys
import gc
########################################################################


########################################################################
# import additional libraries
########################################################################
import numpy as np
import scipy.stats
# from import
from tqdm import tqdm
from sklearn import metrics

try:
    from sklearn.externals import joblib
except:
    import joblib
import torch.optim as optim
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchsummary import summary

########################################################################
# original lib
from Preprocess import common as com
from utils.utils import file_list_to_astdata, generator_ast_format, file_list_to_singleastdata
from Train.aegmm_train import TrainerAEGMM
from datamodule.generate_data import asdData


########################################################################
# load parameter.yaml
########################################################################
yam = "/home/zhaoyi/Project/ASD-AEGMM/ASD/ASDGMM-test.yaml"
param = com.yaml_load(yam)
device_ids = [0, 1]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#######################################################################


########################################################################
# output csv file
########################################################################
def save_csv(save_file_path,
             save_data):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)


########################################################################

########################################################################
# main 01_test.py
########################################################################
if __name__ == "__main__":
    # check mode
    # "development": mode == True
    # "evaluation": mode == False
    mode = com.command_line_chk()
    if mode is None:
        sys.exit(-1)

    # make output result directory
    os.makedirs(param["result_directory"], exist_ok=True)

    # load base directory
    dirs = com.select_dirs(param=param, mode=mode)
    train_dirs = com.select_dirs(param=param, mode='--dev')

    # initialize lines in csv for AUC and pAUC
    csv_lines = []

    if mode:
        performance_over_all = []

    # loop of the base directory
    for idx, target_dir in enumerate(dirs):
        train_dir = train_dirs[idx]
        print("\n===========================")
        print("[{idx}/{total}] {target_dir}".format(target_dir=target_dir, idx=idx + 1, total=len(dirs)))
        machine_type = os.path.split(target_dir)[1]

        print("============== MODEL LOAD ==============")
        # load model file
        history_img = "{img}/history_{machine_type}.png".format(img=param["aegmm"]["img_dir"],
                                                                machine_type=machine_type)
        model_file = "{model}/model_{machine_type}_{epoch}_aegmm.pth".format(model=param["aegmm"]["save_path"],
                                                                                  machine_type=machine_type,
                                                                                  epoch=param["aegmm"]["num_epochs"])
        if not os.path.exists(model_file):
            com.logger.error("{} model not found ".format(machine_type))
            sys.exit(-1)

        ########################################################################
        # getting GMM parameters through training data
        ########################################################################
        files_train, y_true_train = com.file_list_generator(target_dir=train_dir,
                                                            section_name="*",
                                                            dir_name="train",
                                                            mode='--dev')

        feats_train = file_list_to_astdata(files_train,
                                           msg="generate train_dataset",
                                           n_mels=param["feature-ast"]["n_mels"],
                                           n_frames=param["feature-ast"]["n_frames"],
                                           n_hop_frames=param["feature-ast"]["n_hop_frames"],
                                           n_fft=param["feature-ast"]["n_fft"])

        feats_train_data = generator_ast_format(feats_train)
        train_data = asdData(feats_train_data, y_true_train)
        train_data = DataLoader(train_data, batch_size=param["aegmm"]["batch_size"],
                                shuffle=True, num_workers=0)
        model = TrainerAEGMM(param["aegmm"], train_data, device, history_img)
        train_phi, train_mu, train_cov, energy_train, labels_train, model_test = model.calculator_parpmeters(
            model_file, train_data)

        ########################################################################
        # getting results and scores
        ########################################################################
        if mode:
            # results for each machine type
            csv_lines.append([machine_type])
            csv_lines.append(["section", "domain", "AUC", "pAUC", "precision", "recall", "F1 score"])
            performance = []

        dir_names = ["source_test", "target_test"]

        for dir_name in dir_names:
            # list machine id
            section_names = com.get_section_names(target_dir, dir_name=dir_name)

            for section_name in section_names:
                # load test file
                files_test, y_true_test = com.file_list_generator(target_dir=target_dir,
                                                        section_name=section_name,
                                                        dir_name=dir_name,
                                                        mode=mode)

                # setup anomaly score file path
                anomaly_score_csv = "{result}/anomaly_score_{machine_type}_{section_name}_{dir_name}.csv".format(
                    result=param["result_directory"],
                    machine_type=machine_type,
                    section_name=section_name,
                    dir_name=dir_name)
                anomaly_score_list = []

                # setup decision result file path
                decision_result_csv = "{result}/decision_result_{machine_type}_{section_name}_{dir_name}.csv".format(
                    result=param["result_directory"],
                    machine_type=machine_type,
                    section_name=section_name,
                    dir_name=dir_name)
                decision_result_list = []

                print("\n============== BEGIN TEST FOR A SECTION ==============")
                y_pred = [0. for k in files_test]
                for file_idx, file_path in tqdm(enumerate(files_test), total=len(files_test)):
                    try:
                        feats_test = file_list_to_singleastdata(file_path,
                                             msg="generate test_dataset",
                                             n_mels=param["feature-ast"]["n_mels"],
                                             n_frames=param["feature-ast"]["n_frames"],
                                             n_hop_frames=param["feature-ast"]["n_hop_frames"],
                                             n_fft=param["feature-ast"]["n_fft"])
                    except:
                        com.logger.error("File broken!!: {}".format(file_path))

                    feats_test = generator_ast_format(feats_test)
                    test_data = asdData(feats_test, y_true_train)
                    test_data = DataLoader(test_data, batch_size=param["aegmm"]["batch_size"],
                                            shuffle=True, num_workers=0)
                    energy_test, threshold = model.eval(model_test, train_phi, train_mu, train_cov,
                                                                          energy_train,
                                                                          labels_train, test_data)

                    y_pred[file_idx] = np.mean(energy_test[0])
                    decision_threshod = np.mean(threshold[0])

                    # store anomaly scores
                    anomaly_score_list.append([os.path.basename(file_path), y_pred[file_idx]])

                    # store decision results
                    if y_pred[file_idx] > decision_threshod:
                        decision_result_list.append([os.path.basename(file_path), 1])
                    else:
                        decision_result_list.append([os.path.basename(file_path), 0])

                # output anomaly scores
                save_csv(save_file_path=anomaly_score_csv, save_data=anomaly_score_list)
                com.logger.info("anomaly score result ->  {}".format(anomaly_score_csv))

                # output decision results
                save_csv(save_file_path=decision_result_csv, save_data=decision_result_list)
                com.logger.info("decision result ->  {}".format(decision_result_csv))

                if mode:
                    # append AUC and pAUC to lists
                    auc = metrics.roc_auc_score(y_true_test, y_pred)
                    p_auc = metrics.roc_auc_score(y_true_test, y_pred, max_fpr=param["max_fpr"])
                    tn, fp, fn, tp = metrics.confusion_matrix(y_true_test, [1 if x > decision_threshod else 0 for x in
                                                                       y_pred]).ravel()
                    prec = tp / np.maximum(tp + fp, sys.float_info.epsilon)
                    recall = tp / np.maximum(tp + fn, sys.float_info.epsilon)
                    f1 = 2.0 * prec * recall / np.maximum(prec + recall, sys.float_info.epsilon)
                    csv_lines.append(
                        [section_name.split("_", 1)[1], dir_name.split("_", 1)[0], auc, p_auc, prec, recall, f1])
                    performance.append([auc, p_auc, prec, recall, f1])
                    performance_over_all.append([auc, p_auc, prec, recall, f1])
                    com.logger.info("AUC : {}".format(auc))
                    com.logger.info("pAUC : {}".format(p_auc))
                    com.logger.info("precision : {}".format(prec))
                    com.logger.info("recall : {}".format(recall))
                    com.logger.info("F1 score : {}".format(f1))

                print("\n============ END OF TEST FOR A SECTION ============")

        if mode:
            # calculate averages for AUCs and pAUCs
            amean_performance = np.mean(np.array(performance, dtype=float), axis=0)
            csv_lines.append(["arithmetic mean", ""] + list(amean_performance))
            hmean_performance = scipy.stats.hmean(
                np.maximum(np.array(performance, dtype=float), sys.float_info.epsilon), axis=0)
            csv_lines.append(["harmonic mean", ""] + list(hmean_performance))
            csv_lines.append([])


    if mode:
        csv_lines.append(["", "", "AUC", "pAUC", "precision", "recall", "F1 score"])
        # calculate averages for AUCs and pAUCs
        amean_performance = np.mean(np.array(performance_over_all, dtype=float), axis=0)
        csv_lines.append(
            ["arithmetic mean over all machine types, sections, and domains", ""] + list(amean_performance))
        hmean_performance = scipy.stats.hmean(
            np.maximum(np.array(performance_over_all, dtype=float), sys.float_info.epsilon), axis=0)
        csv_lines.append(["harmonic mean over all machine types, sections, and domains", ""] + list(hmean_performance))
        csv_lines.append([])

        # output results
        result_path = "{result}/{file_name}".format(result=param["result_directory"], file_name=param["result_file"])
        com.logger.info("results -> {}".format(result_path))
        save_csv(save_file_path=result_path, save_data=csv_lines)

