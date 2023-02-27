# -*- coding: utf-8 -*-
# @Time    : 2020/7/4 19:18
# @Author  : wenzhang
# @File    : main_DaNN_DJP.py
import torch as tr
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import djp_mmd, data_loader, DaNN, DaNN2
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
from loss_funcs.mmd import *
from loss_funcs.coral import *
from loss_funcs.adv import *
from loss_funcs.lmmd import *
warnings.simplefilter(action='ignore', category=RuntimeWarning)
import matplotlib as mpl
import matplotlib.pyplot as plt
import MMD2
import mmd
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
LAMBDA = 0.25
writer = SummaryWriter("logs_2")
# para of the loss function
# accommodate small values of MMD gradient compared to NNs for each iteration
GAMMA = 10000  # 1000 more weight to transferability
SIGMA = 1  # default 1
def mycrossentropy(y_true, y_pred, e=0.1):
    loss1 = K.categorical_crossentropy(y_true, y_pred)
    loss2 = K.categorical_crossentropy(K.ones_like(y_pred)/2, y_pred)
    return (1-e)*loss1 + e*loss2
''' focal loss '''
def binary_focal_loss(y_true, y_pred,gamma=2, alpha=0.9):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
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
def mmd_loss1(x_src, x_tar):
    MMD = MMD2.MMDLoss()
    return MMD(source=x_src, target=x_tar)
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

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    多核或单核高斯核矩阵函数，根据输入样本集x和y，计算返回对应的高斯核矩阵
    Params:
     source: (b1,n)的X分布样本数组
     target:（b2，n)的Y分布样本数组
     kernel_mul: 多核MMD，以bandwidth为中心，两边扩展的基数，比如bandwidth/kernel_mul, bandwidth, bandwidth*kernel_mul
     kernel_num: 取不同高斯核的数量
     fix_sigma: 是否固定，如果固定，则为单核MMD
    Return:
      sum(kernel_val): 多个核矩阵之和
    '''
     # 堆叠两组样本，上面是X分布样本，下面是Y分布样本，得到（b1+b2,n）组总样本
    n_samples = int(source.shape[0])+int(target.shape[0])
    total = np.concatenate((source, target), axis=0)
    # 对总样本变换格式为（1,b1+b2,n）,然后将后两维度数据复制到新拓展的维度上（b1+b2，b1+b2,n），相当于按行复制
    total0 = np.expand_dims(total,axis=0)
    total0= np.broadcast_to(total0,[int(total.shape[0]), int(total.shape[0]), int(total.shape[1])])
    # 对总样本变换格式为（b1+b2,1,n）,然后将后两维度数据复制到新拓展的维度上（b1+b2，b1+b2,n），相当于按复制
    total1 = np.expand_dims(total,axis=1)
    total1=np.broadcast_to(total1,[int(total.shape[0]), int(total.shape[0]), int(total.shape[1])])
    # total1 - total2 得到的矩阵中坐标（i,j, :）代表total中第i行数据和第j行数据之间的差
    # sum函数，对第三维进行求和，即平方后再求和，获得高斯核指数部分的分子，是L2范数的平方
    L2_distance_square = np.cumsum(np.square(total0-total1),axis=2)
    #调整高斯核函数的sigma值
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = np.sum(L2_distance_square) / (n_samples**2-n_samples)
    # 多核MMD
    #以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    print(bandwidth_list)
    #高斯核函数的数学表达式
    kernel_val = [np.exp(-L2_distance_square / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    #得到最终的核矩阵
    return sum(kernel_val)#多核合并

def MK_MMD(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    计算源域数据和目标域数据的MMD距离
    Params:
     source: (b1,n)的X分布样本数组
     target:（b2，n)的Y分布样本数组
     kernel_mul: 多核MMD，以bandwidth为中心，两边扩展的基数，比如bandwidth/kernel_mul, bandwidth, bandwidth*kernel_mul
     kernel_num: 取不同高斯核的数量
     fix_sigma: 是否固定，如果固定，则为单核MMD
 Return:
     loss: MK-MMD loss
    '''
    batch_size = int(source.shape[0])#一般默认为源域和目标域的batchsize相同
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    # 将核矩阵分成4部分
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i + 1) % batch_size
        t1, t2 = s1 + batch_size, s2 + batch_size
        loss += kernels[s1, s2] + kernels[t1, t2]
        loss -= kernels[s1, t2] + kernels[s2, t1]
    # 这里计算出的n_loss是每个维度上的MK-MMD距离，一般还会做均值化处理
    n_loss= loss / float(batch_size)
    return np.mean(n_loss)

def kl_divergence(p,q):
    return sum(p[i] * math.log2(p[i] / q[i]) for i in range(len(p)))

def js_divergence(p, q):
	m = 0.5 * (p + q)
	return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

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
        x_src, y_src = x_src.detach().view(-1, 11).to(DEVICE), y_src.to(DEVICE)
        x_tar, y_tar = x_tar.detach().view(-1, 11).to(DEVICE), y_tar.to(DEVICE)
        model.train()
        ypred, x_src_mmd, x_tar_mmd = model(x_src, x_tar)

        # print('x_src: ', x_src.shape, '\t x_tar', x_tar.shape)  # both torch.Size([64, 784])
        loss_ce = criterion(ypred, y_src)
        # loss_ce = binary_focal_loss(ypred, y_src)
        # loss_mmd = mmd_loss(x_src_mmd, y_src, x_tar_mmd, y_pse, mmd_type)
        # loss_mmd = CORAL(x_src_mmd, x_tar_mmd)
        # loss_mmd = mmd_loss1(x_src_mmd,x_tar_mmd)
        # loss_mmd = mmd_loss1(x_src,x_tar)
        # # x_src_mmd = np.float32(x_src_mmd.detach().numpy())
        # # x_tar_mmd = np.float32(x_tar_mmd.detach().numpy())
        loss_mmd = MK_MMD(x_src,x_tar)
        # loss_mmd = kl_divergence(x_src,x_tar)
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
        # print(' ')
        # print(matrix)

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
        tmp_train_loss += loss.detach()
        batch_j += 1
        if batch_j >= len(list_tar):
            batch_j = 0

    total_G_mean_train /=  len(data_src)
    tmp_train_loss /= len(data_src)
    tmp_train_acc = correct * 100. / len(data_src.dataset)
    train_loss = tmp_train_loss.detach().cpu().numpy()
    train_acc = tmp_train_acc.numpy()
    # total_G_mean_train = total_G_mean_train.detach().cpu().numpy()

    tim = time.strftime("%H:%M:%S", time.localtime())
    res_e = '{:s}, epoch: {}/{}, train loss: {:.4f}, train acc: {:.4f}, train G_mean: {:.4f}'.format(
        tim, epoch, N_EPOCH, tmp_train_loss, tmp_train_acc, total_G_mean_train)
    tqdm.write(res_e)
    # print('G_mean_train:', total_G_mean_train)
    # log_train.write(res + '\n')
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


def model_test(model, data_tar, e):
    tmp_test_loss = 0
    correct = 0
    total_G_mean_test = 0
    criterion = nn.CrossEntropyLoss()
    with tr.no_grad():
        for batch_id, (x_tar, y_tar) in enumerate(data_tar):
            x_tar, y_tar = x_tar.detach().view(-1, 11).to(DEVICE), y_tar.to(DEVICE)
            model.eval()
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
            total_G_mean_test += G_mean_test
            # total_G_mean_test.append(G_mean_test)
            #'''
            tmp_test_loss += loss.detach()

        total_G_mean_test /= len(data_tar)
        tmp_test_loss /= len(data_tar)
        tmp_test_acc = correct * 100. / len(data_tar.dataset)
        test_loss = tmp_test_loss.detach().cpu().numpy()
        test_acc = tmp_test_acc.detach().cpu().numpy()
        # total_G_mean_test = total_G_mean_test.detach().cpu().numpy()

        res = 'test loss: {:.4f}, test acc: {:.4f}, test G_mean: {:.4f}'.format(tmp_test_loss, tmp_test_acc, total_G_mean_test)
    tqdm.write(res)
    # log_test.write(res + '\n')
    # print('G_mean_test:', G_mean_test)
    return test_acc, test_loss, total_G_mean_test


def main():
    tr.manual_seed(1)
    # dataset = pd.read_csv("data_year/ST4000DM000.csv")
    dataset = pd.read_csv("ST4000_86/ST12000NM0007.csv")
    X_train = dataset.drop(['ID', 'data', 'serial_number', 'model', 'capacity_bytes', 'failure'], axis=1).values
    Y_train = dataset['failure'].values
    encoder = LabelEncoder()
    Y_train = encoder.fit_transform(Y_train.ravel())

    X_train, Y_train = torch.FloatTensor(X_train), torch.LongTensor(Y_train)
    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    X_s = DataLoader(dataset=train_dataset, batch_size=3000, shuffle=True)

    dataset = pd.read_csv("data_year/ST4000DM000.csv")
    X_test = dataset.drop(['ID', 'date', 'serial_number', 'model', 'capacity_bytes', 'failure'], axis=1).values
    Y_test = dataset['failure'].values
    encoder = LabelEncoder()
    Y_test = encoder.fit_transform(Y_test.ravel())
    X_test, Y_test = torch.FloatTensor(X_test), torch.LongTensor(Y_test)
    # X_test1, X_test2, Y_test1, Y_test2 = train_test_split(X_test, Y_test, test_size=0.5, stratify=Y_test)
    test_dataset1 = torch.utils.data.TensorDataset(X_test, Y_test)
    X_t = DataLoader(dataset=test_dataset1, batch_size=3000, shuffle=True)
    # test_dataset2 = torch.utils.data.TensorDataset(X_test2, Y_test2)
    # X_t2 = DataLoader(dataset=test_dataset2, batch_size=3000, shuffle=False)

    # rootdir = "office_caltech_10/"
    # tr.manual_seed(1)
    # domain_str = ['webcam', 'dslr']
    # X_s = data_loader.load_train(root_dir=rootdir, domain=domain_str[0], batch_size=BATCH_SIZE[0])
    # X_t = data_loader.load_test(root_dir=rootdir, domain=domain_str[1], batch_size=BATCH_SIZE[1])

    # train and test
    start = time.time()
    mmd_type = ['jpmmd']       #'mmd','jmmd','jpmmd','djpmmd'
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
        max = 0
        epoch = 0
        y_pse = tr.zeros(64, 64).long()         #返回一个形状为为size,类型为torch.dtype，里面的每一个值都是0的tensor

        mdl = DaNN2.DaNN2(n_input=11, n_hidden=256, n_class=2)
        #64X   128:train 0.64  test:0.36  256:train：  0.73 test:0.53  512:train  0.75test:0.41
        mdl = mdl.to(DEVICE)

        # optimization
        opt_Adam = optim.Adam(mdl.parameters(), lr=LEARNING_RATE)

        for ep in tqdm(range(1, N_EPOCH + 1)):
            tmp_train_acc, tmp_train_loss, train_G_mean, mdl = \
                model_train(model=mdl, optimizer=opt_Adam, epoch=ep, data_src=X_s, data_tar=X_t, y_pse=y_pse,
                            mmd_type=mt)        #
            tmp_test_acc, tmp_test_loss, test_G_mean = model_test(mdl, X_t, ep)  #
            # train_acc.append(tmp_train_acc)
            # train_loss.append(tmp_train_loss)
            # test_acc.append(tmp_test_acc)
            # train_G_mean.append(train_G_mean)
            # test_G_mean.append(test_G_mean)
            # test_loss.append(tmp_test_loss)
            # writer.add_scalar('train_G_mean', train_G_mean, ep + 1)
            # writer.add_scalar('test_G_mean', test_G_mean, ep + 1)
            if test_G_mean > max:
                max = test_G_mean
                epoch = ep
        # torch.save(mdl, 'model_dann2.pkl')
        print(max)
        print(epoch)


        acc['train'], acc['test'] = train_acc, test_acc
        G_mean['train'],G_mean['test'] = train_G_mean, test_G_mean
        loss['train'], loss['test'] = train_loss, test_loss

        # visualize
        plt.plot(G_mean['train'], label='train-' + mt)
        plt.plot(G_mean['test'], label='test-' + mt, ls='--')

    # plt.title(domain_str[0] + ' to ' + domain_str[1])
    plt.xticks(np.linspace(1, N_EPOCH, num=2, dtype=np.int8))
    plt.xlim(1, N_EPOCH)
    plt.ylim(0, 100)
    plt.legend(loc='upper right')
    plt.xlabel("epochs")
    plt.ylabel("G-mean")
    # plt.savefig(domain_str[0] + '_' + domain_str[1] + "_acc.jpg")
    plt.close()

    # time and save model
    end = time.time()
    print("Total run time: %.2f" % float(end - start))


if __name__ == '__main__':
    main()
