# -*- coding: utf-8 -*-
# @Time    : 2020/7/4 19:18
# @Author  : wenzhang
# @File    : main_DaNN_DJP.py
import torch as tr
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import djp_mmd, data_loader, DaNN
import time
import math
import random
import warnings
import numpy as np
import torch.optim
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

warnings.simplefilter(action='ignore', category=RuntimeWarning)
import matplotlib as mpl
import matplotlib.pyplot as plt
import MMD2
import os
import tensorflow as tf
import keras.backend as K

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = tr.device('cuda' if tr.cuda.is_available() else 'cpu')

# para of the network
LEARNING_RATE = 0.001  # 0.001
DROPOUT = 0.5
N_EPOCH = 10000
BATCH_SIZE = [64, 64]  # bathsize of source and target domain

# para of the loss function
# accommodate small values of MMD gradient compared to NNs for each iteration
GAMMA = 1000  # 1000 more weight to transferability
SIGMA = 1  # default 1

''' focal loss '''

def binary_focal_loss(y_true, y_pred,gamma=2, alpha=0.9):
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    # def binary_focal_loss_fixed(y_true, y_pred):
    """
    y_true shape need be (None,1)
    y_pred need be compute after sigmoid
    """
    y_true = tf.cast(y_true, tf.float32)
    alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)

    p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()
    focal_loss = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
    return K.mean(focal_loss)

    # return binary_focal_loss_fixed
# MMD, JMMD, JPMMD, DJP-MMD
def mmd_loss(x_src, y_src, x_tar, y_pseudo, mmd_type):
    if mmd_type == 'mmd':
        return djp_mmd.rbf_mmd(x_src, x_tar, SIGMA)
    elif mmd_type == 'jmmd':
        return djp_mmd.rbf_jmmd(x_src, y_src, x_tar, y_pseudo, SIGMA)
    elif mmd_type == 'jpmmd':
        return djp_mmd.rbf_jpmmd(x_src, y_src, x_tar, y_pseudo, SIGMA)
    elif mmd_type == 'djpmmd':
        return djp_mmd.rbf_djpmmd(x_src, y_src, x_tar, y_pseudo, SIGMA)

def model_train(model, optimizer, epoch, data_src, data_tar, y_pse, mmd_type):
    tmp_train_loss = 0
    correct = 0
    batch_j = 0
    total_G_mean_train = 0
    criterion = nn.CrossEntropyLoss()
    list_src, list_tar = list(enumerate(data_src)), list(enumerate(data_tar))


    # print('***********', len(list_src), len(list_tar))
    for batch_id, (x_src, y_src) in enumerate(data_src):
        optimizer.zero_grad()
        _, (x_tar, y_tar) = list_tar[batch_j]
        model.train()
        ypred, x_src_mmd, x_tar_mmd = model(x_src, x_tar)

        # print('x_src: ', x_src.shape, '\t x_tar', x_tar.shape)  # both torch.Size([64, 784])
        loss_ce = criterion(ypred, y_src)
        loss_mmd = mmd_loss(x_src_mmd, y_src, x_tar_mmd, y_pse, mmd_type)
        # loss_mmd = MMD.mmd(x_src_mmd,x_tar_mmd)
        # pred = ypred.detach().max(1)[1]  # get the index of the max log-probability
        pred = ypred.argmax(axis=1)
        loss = loss_ce + GAMMA * loss_mmd
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # get pseudo labels of the target
        # model.eval()
        # pred_pse, _, _ = model(x_tar, x_tar)
        # y_pse = pred_pse.argmax(axis=1)

        # get training loss
        correct += pred.eq(y_src.detach().view_as(pred)).cpu().sum()
        #'''

        matrix = confusion_matrix(y_src, pred)
        print(' ')
        print(matrix)

        TN = matrix[0][0]
        # print(TN)
        FP = matrix[0][1]
        # print(FP)
        FN = matrix[1][0]
        # print(FN)
        TP = matrix[1][1]
        # print(TP)
        FDR = TP / (TP + FN)
        # print(FDR)
        P = TN / (TN + FP)
        # print(P)
        if np.isnan(FDR):
            FDR = 0.0
        if np.isnan(P):
            P = 0.0
        G_mean_train = math.sqrt(FDR * P)
        print(G_mean_train)
        # print(type(total_G_mean_train))
        # print(type(G_mean_train))
        if np.isnan(G_mean_train):
            G_mean_train = 0.0
        total_G_mean_train += G_mean_train
        # total_G_mean_train.append(G_mean_train)
        #'''
        # loss = loss_ce + GAMMA * loss_mmd

        # error backward
        # loss.backward()
        # optimizer.step()
        tmp_train_loss += loss.detach()

    total_G_mean_train = total_G_mean_train / len(data_src)
    tmp_train_loss /= len(data_src)
    tmp_train_acc = correct * 100. / len(data_src.dataset)
    train_loss = tmp_train_loss.detach().cpu().numpy()
    train_acc = tmp_train_acc.numpy()
    # total_G_mean_train = total_G_mean_train.detach().cpu().numpy()

    tim = time.strftime("%H:%M:%S", time.localtime())
    res_e = '{:s}, epoch: {}/{}, train loss: {:.4f}, train acc: {:.4f}'.format(
        tim, epoch, N_EPOCH, tmp_train_loss, tmp_train_acc)
    tqdm.write(res_e)
    print('G_mean_train:', total_G_mean_train)
    return train_acc, train_loss, total_G_mean_train, model#

def model_train1(model, optimizer, epoch, data_src, data_tar, y_pse, mmd_type):
    tmp_train_loss = 0
    correct = 0
    batch_j = 0
    total_G_mean_train=0
    criterion = nn.CrossEntropyLoss()
    list_src, list_tar = list(enumerate(data_src)), list(enumerate(data_tar))

    # print('***********', len(list_src), len(list_tar))
    for batch_id, (x_src, y_src) in enumerate(data_src):
        optimizer.zero_grad()
        x_src, y_src = x_src.detach().view(-1, 11).to(DEVICE), y_src.to(DEVICE)
        _, (x_tar, y_tar) = list_tar[batch_j]
        x_tar = x_tar.view(-1, 11).to(DEVICE)
        model.train()
        ypred, x_src_mmd, x_tar_mmd = model(x_src, x_tar)

        # print('x_src: ', x_src.shape, '\t x_tar', x_tar.shape)  # both torch.Size([64, 784])
        loss_ce = criterion(ypred, y_src)
        loss_mmd = mmd_loss(x_src_mmd, y_src, x_tar_mmd, y_pse[batch_id, :], mmd_type)
        # pred = ypred.detach().max(1)[1]  # get the index of the max log-probability
        pred = ypred.argmax(axis=1)

        # get pseudo labels of the target
        model.eval()
        pred_pse, _, _ = model(x_tar, x_tar)
        y_pse[batch_id, :] = pred_pse.detach().max(1)[1]

        # get training loss
        correct += pred.eq(y_src.detach().view_as(pred)).cpu().sum()
        #'''

        matrix = confusion_matrix(y_src, pred)

        TN = matrix[0][0]
        FP = matrix[0][1]
        FN = matrix[1][0]
        TP = matrix[1][1]
        FDR = TP / (TP + FN)
        FAR = FP / (FP + TN)
        P = TN / (TN + FP)
        G_mean_train = math.sqrt(FDR * P)
        if np.isnan(G_mean_train):
            G_mean_train = 0.0
        total_G_mean_train += G_mean_train
        #'''
        loss = loss_ce + GAMMA * loss_mmd

        # error backward
        loss.backward()
        optimizer.step()
        tmp_train_loss += loss.detach()

    total_G_mean_train = total_G_mean_train / len(data_src)
    tmp_train_loss /= len(data_src)
    tmp_train_acc = correct * 100. / len(data_src.dataset)
    train_loss = tmp_train_loss.detach().cpu().numpy()
    train_acc = tmp_train_acc.numpy()
    train_G_mean = total_G_mean_train.detach().cpu().numpy()

    tim = time.strftime("%H:%M:%S", time.localtime())
    res_e = '{:s}, epoch: {}/{}, train loss: {:.4f}, train acc: {:.4f}'.format(
        tim, epoch, N_EPOCH, tmp_train_loss, tmp_train_acc)
    tqdm.write(res_e)
    return train_acc, train_loss, train_G_mean, model#


def model_test(model, data_tar):
    model.eval()
    tmp_test_loss = 0
    correct = 0
    total_G_mean_test = []
    criterion = nn.CrossEntropyLoss()
    with tr.no_grad():
        for batch_id, (x_tar, y_tar) in enumerate(data_tar):
            x_tar, y_tar = x_tar.to(DEVICE), y_tar.to(DEVICE)
            ypred, _, _ = model(x_tar, x_tar)
            loss = criterion(ypred, y_tar)
            pred = ypred.detach().max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(y_tar.detach().view_as(pred)).cpu().sum()
            #'''
            matrix = confusion_matrix(y_tar, pred)
            print(' ')
            print(matrix)
            TN = matrix[0][0]
            FP = matrix[0][1]
            FN = matrix[1][0]
            TP = matrix[1][1]
            FDR = TP / (TP + FN)
            FAR = FP / (FP + TN)
            P = TN / (TN + FP)
            G_mean_test = math.sqrt(FDR * P)
            if np.isnan(G_mean_test):
                G_mean_test = 0.0
            # total_G_mean_test += G_mean_test
            total_G_mean_test.append(G_mean_test)
            #'''
            tmp_test_loss += loss.detach()

        # total_G_mean_test = total_G_mean_test / len(data_tar)
        tmp_test_loss /= len(data_tar)
        tmp_test_acc = correct * 100. / len(data_tar.dataset)
        test_loss = tmp_test_loss.detach().cpu().numpy()
        test_acc = tmp_test_acc.detach().cpu().numpy()
        # test_G_mean = total_G_mean_test.detach().cpu().numpy()

        res = 'test loss: {:.4f}, test acc: {:.4f}'.format(tmp_test_loss, tmp_test_acc)
    tqdm.write(res)
    print('G_mean_test:', G_mean_test)
    return test_acc, test_loss, total_G_mean_test


def main():
    dataset = pd.read_csv("data_year/ST4000DM000.csv")
    X_train = dataset.drop(['ID', 'date', 'serial_number', 'model', 'capacity_bytes', 'failure'], axis=1).values
    Y_train = dataset['failure'].values
    encoder = LabelEncoder()
    Y_train = encoder.fit_transform(Y_train.ravel())

    X_train, Y_train = torch.FloatTensor(X_train), torch.LongTensor(Y_train)
    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    X_s = DataLoader(dataset=train_dataset, batch_size=3000, shuffle=False)

    dataset = pd.read_csv("data_year/ST8000NM0055.csv")
    X_test = dataset.drop(['ID', 'date', 'serial_number', 'model', 'capacity_bytes', 'failure'], axis=1).values
    Y_test = dataset['failure'].values
    encoder = LabelEncoder()
    Y_test = encoder.fit_transform(Y_test.ravel())
    X_test, Y_test = torch.FloatTensor(X_test), torch.LongTensor(Y_test)
    X_test1, X_test2, Y_test1, Y_test2 = train_test_split(X_test, Y_test, test_size=0.5, stratify=Y_test)
    test_dataset1 = torch.utils.data.TensorDataset(X_test1, Y_test1)
    X_t = DataLoader(dataset=test_dataset1, batch_size=3000, shuffle=False)
    test_dataset2 = torch.utils.data.TensorDataset(X_test2, Y_test2)
    X_t2 = DataLoader(dataset=test_dataset2, batch_size=3000, shuffle=False)

    # rootdir = "office_caltech_10/"
    # tr.manual_seed(1)
    # domain_str = ['webcam', 'dslr']
    # X_s = data_loader.load_train(root_dir=rootdir, domain=domain_str[0], batch_size=BATCH_SIZE[0])
    # X_t = data_loader.load_test(root_dir=rootdir, domain=domain_str[1], batch_size=BATCH_SIZE[1])

    # train and test
    start = time.time()
    mmd_type = ['jpmmd']       # 'mmd','jmmd',
    for mt in mmd_type:
        # print('-' * 10 + domain_str[0] + ' -->  ' + domain_str[1] + '-' * 10)
        print('MMD loss type: ' + mt + '\n')
        acc, loss, G_mean = {}, {}, {}
        train_acc = []
        test_acc = []
        train_G_mean = []
        test_G_mean = []
        train_loss = []
        test_loss = []
        y_pse = tr.zeros(64, 64).long()         #返回一个形状为为size,类型为torch.dtype，里面的每一个值都是0的tensor

        mdl = DaNN.DaNN(n_input=11, n_hidden=300, n_class=2)
        #64X   128:train 0.64  test:0.36  256:train：  0.73 test:0.53  512:train:0.75  test:0.55  648:train:0.77 test:0.44
        mdl = mdl.to(DEVICE)

        # optimization
        opt_Adam = optim.Adam(mdl.parameters(), lr=LEARNING_RATE)

        for ep in tqdm(range(1, N_EPOCH + 1)):
            tmp_train_acc, tmp_train_loss, train_G_mean, mdl = \
                model_train(model=mdl, optimizer=opt_Adam, epoch=ep, data_src=X_s, data_tar=X_t, y_pse=y_pse,
                            mmd_type=mt)        #
            train_acc.append(tmp_train_acc)
            train_loss.append(tmp_train_loss)
        tmp_test_acc, tmp_test_loss, test_G_mean = model_test(mdl, X_t2)  #
        test_acc.append(tmp_test_acc)
        # train_G_mean.append(train_G_mean)
        # test_G_mean.append(test_G_mean)
        test_loss.append(tmp_test_loss)
        acc['train'], acc['test'] = train_acc, test_acc
        G_mean['train'],G_mean['test'] = train_G_mean, test_G_mean
        loss['train'], loss['test'] = train_loss, test_loss

        # visualize
        plt.plot(acc['train'], label='train-' + mt)
        plt.plot(acc['test'], label='test-' + mt, ls='--')

    # plt.title(domain_str[0] + ' to ' + domain_str[1])
    plt.xticks(np.linspace(1, N_EPOCH, num=2, dtype=np.int8))
    plt.xlim(1, N_EPOCH)
    plt.ylim(0, 100)
    plt.legend(loc='upper right')
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    # plt.savefig(domain_str[0] + '_' + domain_str[1] + "_acc.jpg")
    plt.close()

    # time and save model
    end = time.time()
    print("Total run time: %.2f" % float(end - start))


if __name__ == '__main__':
    main()
