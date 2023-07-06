import numpy as np
import os
from src.FeatureSelector import FeatureSelector
from DataGenerator import generate_sythetic_data, get_one_hot
from sklearn.preprocessing import StandardScaler
from itertools import repeat
from torch.multiprocessing import Pool
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import argparse
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_selection import f_classif, mutual_info_classif
from tensorflow.keras import datasets
import torch
import cv2
import warnings
warnings.filterwarnings('ignore')

def plot_feature_importance(importances_all, feature_selected_all, fig_name):

    fontsize = 15
    importance_mean = np.array(importances_all).mean(axis=0).flatten()

    count = pd.value_counts(np.concatenate((feature_selected_all)))
    count_index = np.array(count.index)
    count_values = np.array(count.values)
    count_values_sorted = count_values[np.argsort(count_index)]
    count_index_sorted = count_index[np.argsort(count_index)]
    complete_index = []
    complete_values = []

    j = 0
    for i in range(10):
        complete_index.append(i)
        if i in count_index_sorted:
            complete_values.append(count_values_sorted[j])
            j = j + 1
        else:
            complete_values.append(0)

    colours = []
    for i in range(10):
        if complete_values[i] == args.n_folds:
            colours.append('r')
        else:
            colours.append('g')

    plt.bar(x=range(10), height=importance_mean, align="center", color=colours)
    plt.ylabel('Feature importance', fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xticks(range(10), fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(fig_name, dpi=600)
    plt.show()

def save_feature_map(importances_all, base_name):

    plt.figure()

    # save feature importance map
    importances_ = np.array(importances_all).mean(axis=0)
    importances_ = normalize_0_1(importances_)
    importances_reshaped = np.reshape(importances_, (28, 28))
    sc = plt.imshow(importances_reshaped)
    plt.axis('off')
    plt.savefig(base_name + '_' + 'feature_importance_map1.jpg', dpi=600, bbox_inches='tight', pad_inches=0)
    sc.set_cmap('hot')
    plt.savefig(base_name + '_' + 'feature_importance_map2.jpg', dpi=600, bbox_inches='tight', pad_inches=0)

    # save the mean images of digit ‘3’
    digit3_image = normalize_0_1(x_train[digit3_index, :].mean(axis=0))
    sc = plt.imshow(digit3_image)
    sc.set_cmap('Greens')
    plt.axis('off')
    plt.savefig("pictures/" + 'digit3_mean.jpg', dpi=600, bbox_inches='tight', pad_inches=0)

    # save the mean images of digit ‘8’
    digit3_image = normalize_0_1(x_train[digit8_index, :].mean(axis=0))
    sc = plt.imshow(digit3_image)
    sc.set_cmap('Greens')  # Greens, Blues, Purples
    plt.axis('off')
    plt.savefig("pictures/" + 'digit8_mean.jpg', dpi=600, bbox_inches='tight', pad_inches=0)

    # Feature importance map is superimposed on the mean images of digit ‘3’
    img1 = cv2.addWeighted(cv2.imread(base_name + '_' + 'feature_importance_map2.jpg'), 0.5, cv2.imread("pictures/" + 'digit3_mean.jpg'), 0.5, 0)
    plt.imshow(img1)
    plt.axis('off')
    plt.savefig(base_name + '_' + 'digit3_mean_plus_feature_importance_map.jpg', dpi=600, bbox_inches='tight', pad_inches=0)

    # Feature importance map is superimposed on the mean images of digit ‘8’
    img2 = cv2.addWeighted(cv2.imread(base_name + '_' + 'feature_importance_map2.jpg'), 0.5, cv2.imread("pictures/" + 'digit8_mean.jpg'), 0.5, 0)
    plt.imshow(img2)
    plt.axis('off')
    plt.savefig(base_name + '_' + 'digit8_mean_plus_feature_importance_map.jpg', dpi=600, bbox_inches='tight', pad_inches=0)

    plt.subplot(1, 3, 1)
    img1 = cv2.imread(base_name + '_' + 'feature_importance_map1.jpg')
    plt.imshow(img1[:, :, [2, 1, 0]])
    plt.axis('off')

    plt.subplot(1, 3, 2)
    img2 = cv2.imread(base_name + '_' + 'digit3_mean_plus_feature_importance_map.jpg')
    plt.imshow(img2[:, :, [2, 1, 0]])
    plt.axis('off')

    plt.subplot(1, 3, 3)
    img3 = cv2.imread(base_name + '_' + 'digit8_mean_plus_feature_importance_map.jpg')
    plt.imshow(img3[:, :, [2, 1, 0]])
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(base_name + '_' + 'feature_importance.jpg', dpi=600, bbox_inches="tight")
    # plt.show()

def mutual_info(trn_feats, trn_labels, fs):

    importances = mutual_info_classif(trn_feats, trn_labels, random_state=0)
    index_selected = np.argsort(-importances)[0:fs]

    return trn_feats[:, index_selected], index_selected, importances

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def classifier(kernel, C):
    return SVC(kernel=kernel, C=C, probability=True, gamma='auto', random_state=0)
def classification(X_train, y_train, X_test, y_test, args):
    clf = classifier(args.kernel_type, args.SVM_C)
    clf.fit(X_train, y_train)
    pre_dict_label = clf.predict(X_test)
    predict_pro_tst = clf.predict_proba(X_test)
    return pre_dict_label, predict_pro_tst

def Standardize(X_train, X_test):
    std = StandardScaler().fit(X_train)
    X_train = std.transform(X_train)
    X_test = std.transform(X_test)
    return X_train, X_test

def cross_val_index_split(trn_feats, n_folds):
    cv_outer = KFold(n_splits=n_folds, shuffle=True, random_state=0)
    train_index_all = []
    test_index_all = []
    for train_index, test_index in cv_outer.split(trn_feats):
        train_index_all.append(train_index)
        test_index_all.append(test_index)
    return train_index_all, test_index_all

def get_nfs_top_rank_index(rank, nfs):
    rank_index = np.argsort(-rank)
    return rank_index[0: nfs]

def cross_val_run(X, y, train_ix, test_ix, args):

    X_train, X_test = X[train_ix, :], X[test_ix, :]
    y_train, y_test = y[train_ix], y[test_ix]
    X_train, X_test = Standardize(X_train, X_test)
    importances = DFR_run(X_train, y_train, args)

    feature_selected = get_nfs_top_rank_index(importances, args.num_fs)          # The selected features

    predicted_label, _ = classification(X_train[:, feature_selected], y_train, X_test[:, feature_selected], y_test, args) # Classification
    ACC = accuracy_score(y_test, predicted_label)

    return ACC, importances, feature_selected

def cross_validation(X, y, args):

    importances_all = []
    ACC_all = []
    feature_selected_all = []

    train_index_all, test_index_all = cross_val_index_split(X, args.n_folds)

    if args.multi_thread:
        # Run folds in multi-threading
        thread_agrs = list(zip(repeat(X), repeat(y), train_index_all, test_index_all, repeat(args)))
        pool = Pool(args.num_workers)
        results = pool.starmap(cross_val_run, thread_agrs)
        for result in results:
            ACC_all.append(result[0])
            importances_all.append(result[1])
            feature_selected_all.append(result[2])
    else:
        # Run each fold
        for i in range(len(train_index_all)):
            ACC, importances, feature_selected = cross_val_run(X, y, train_index_all[i], test_index_all[i], args)
            ACC_all.append(ACC)
            importances_all.append(importances)
            feature_selected_all.append(feature_selected)

    return ACC_all, importances_all, feature_selected_all

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def DFR_run(train_features, train_labels, args):

    torch.set_default_dtype(torch.float64)

    data_batch_size = args.data_batch_size
    mask_batch_size = args.mask_batch_size
    phase_2_start = args.phase_2_start
    max_batches = args.max_batches

    operator_arch = args.operator_arch
    selector_arch = args.selector_arch

    s = args.s
    s_p = args.s_p

    epoch_on_which_selector_trained = args.epoch_on_which_selector_trained

    FEATURE_SHAPE = (train_features.shape[1],)

    train_labels = get_one_hot(train_labels.astype(np.int8), len(np.unique(train_labels)))  # Get one-hot labels

    fs = FeatureSelector(FEATURE_SHAPE, s, data_batch_size, mask_batch_size,
                         epoch_on_which_selector_trained=epoch_on_which_selector_trained)

    fs.create_dense_operator(operator_arch)

    fs.create_dense_selector(selector_arch)

    fs.create_mask_optimizer(epoch_condition=phase_2_start, perturbation_size=s_p)

    fs.train_networks_on_data(train_features, train_labels, max_batches, args)

    importances = fs.get_importances()

    return importances

def normalize_0_1(importances_):
    return (importances_ - importances_.min()) / (importances_.max() - importances_.min())

if __name__ == '__main__':

    # python DFR_main.py --run_example1 --operator_arch 128 32 4 --num_fs 5 --multi_thread
    # python DFR_main.py --run_example2 --operator_arch 128 32 2 --num_fs 5 --multi_thread
    # python DFR_main.py --run_example3 --num_fs 50 --s 50 --s_p 20 --multi_thread
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_example1', action="store_true")              # XOR synthetic dataset classification
    parser.add_argument('--run_example2', action="store_true")              # binary classification
    parser.add_argument('--run_example3', action="store_true")              # Mnist hand-written digit feature importance visulization
    parser.add_argument('--num_training_samples', type=int, default=2500)   # Number of training samples
    parser.add_argument('--num_fs', type=int, default=5)                    # Number of selected features in each fold
    parser.add_argument('--n_folds', type=int, default=5)                   # Number of folds in cross-validations
    parser.add_argument('--kernel_type', type=str, default="rbf")           # Kernel type in SVM
    parser.add_argument('--SVM_C', type=float, default=1.0)                 # C in SVM
    parser.add_argument('--multi_thread', action="store_true")              # Run in multi-threading
    parser.add_argument('--num_workers', type=int, default=5)               # Number of workers in multi-threading
    parser.add_argument('--class_weighted', action="store_true")
    # parameter for DFR
    parser.add_argument('--data_batch_size', type=int, default=32)                        # Batch size
    parser.add_argument('--mask_batch_size', type=int, default=32)                        # The size of the feature mask subset, e.g., |Z|
    parser.add_argument('--s', type=int, default=5)                                       
    parser.add_argument('--s_p', type=int, default=2)                                     
    parser.add_argument('--phase_2_start', type=int, default=6000)
    parser.add_argument('--max_batches', type=int, default=10000)                         # Number of iterations
    parser.add_argument('--epoch_on_which_selector_trained', type=int, default=2)
    parser.add_argument('--operator_arch', nargs='+', type=int, default=[128, 32, 4])     # Operator's architecture
    parser.add_argument('--selector_arch', nargs='+', type=int, default=[128, 32, 1])     # Selector's architecture

    parser.add_argument('--CUDA_VISIBLE_DEVICES', nargs='+', type=str, default="0")
    parser.add_argument('--seed', type=int, default=8888)                                 # seed
    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn')

    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    for CUDA_VISIBLE_DEVICES in args.CUDA_VISIBLE_DEVICES:
        os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
    seed_everything(args.seed)

    if args.run_example1:
        # XOR classification
        X_tr, y_tr = generate_sythetic_data(args.num_training_samples, 'XOR', args.seed)

        ACC_all, importances_all, feature_selected_all = cross_validation(X_tr, y_tr, args)
        print("Cross-validation ACC mean and std: %.3f %.3f" % (np.mean(ACC_all), np.std(ACC_all)))

        plot_feature_importance(importances_all, feature_selected_all, 'results/XOR_classification.pdf')

    if args.run_example2:
        # binary classification
        X_tr, y_tr = generate_sythetic_data(args.num_training_samples, 'binary_classification', args.seed)

        ACC_all, importances_all, feature_selected_all = cross_validation(X_tr, y_tr, args)
        print("Cross-validation ACC mean and std: %.3f %.3f" % (np.mean(ACC_all), np.std(ACC_all)))

        plot_feature_importance(importances_all, feature_selected_all, 'results/binary_classification.pdf')

    if args.run_example3:
        # Mnist hand-written digit feature importance visulization
        (x_train, y_train), (_, _) = datasets.mnist.load_data()
        digit3_index = np.where(y_train == 3)[0]
        digit8_index = np.where(y_train == 8)[0]

        train_data = np.concatenate((x_train[digit3_index, :], x_train[digit8_index, :]), axis=0).reshape(-1, 784).astype(np.float32) / 255
        train_labels = np.concatenate((np.zeros(shape=digit3_index.shape), np.ones(shape=digit8_index.shape)))

        # kepp the same network architecture for the operator
        args.operator_arch = [32, 32, 2]

        # Use 3 different selector architectures for 3 different runs
        selector_archs = [[1], [32, 1], [32, 32, 1]]
        acc_all = []
        std_all = []
        base_name_all = []

        for selector_arch in selector_archs:

            args.selector_arch = selector_arch

            # run cross validation
            ACC_all, importances_all, feature_selected_all = cross_validation(train_data, train_labels, args)
            print("Operator architecture:", args.operator_arch, "Selector architecture:", args.selector_arch)
            print("Cross-validation ACC mean and std: %.3f %.3f" % (np.mean(ACC_all), np.std(ACC_all)))

            # append accuracy
            acc_all.append(np.mean(ACC_all))
            std_all.append(np.std(ACC_all))

            # save path of the feature map
            base_name = "pictures/" + "selector_arch" + '_784_' + '_'.join(str(d) for d in args.selector_arch)
            base_name_all.append(base_name)

            # save feature map
            save_feature_map(importances_all, base_name)

        print("")
        print("left:   The feature importance map, \n"
              "middle: The feature importance map uperimposed on the mean images of digit ‘3’, \n"
              "right:  The feature importance map uperimposed on the mean images of digit ‘8’")

        # show images:
        plt.figure()
        for i, selector_arch in enumerate(selector_archs):
            plt.subplot(3, 1, i+1)
            img = cv2.imread(base_name_all[i] + '_' + 'feature_importance.jpg')
            selector_arch.insert(0, 784)
            plt.title("Selector architecture: " + str(selector_arch) + ", ACC: %.3f %.3f" % (acc_all[i], std_all[i]), fontsize=9)
            plt.imshow(img[:, :, [2, 1, 0]])
            plt.axis("off")

        plt.tight_layout()
        plt.savefig("results/" + 'feature_importance.pdf', dpi=600, bbox_inches="tight")
        plt.show()
