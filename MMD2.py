'''
import numpy as np
import pandas as pd
import scipy.stats
import torch

p=np.asarray([0.65,0.25,0.07,0.03])
q=np.array([0.6,0.25,0.1,0.05])
def KL_divergence(p,q):
    return scipy.stats.entropy(p, q, base=2)
# print(KL_divergence(p,q)) # 0.01693110139988926
# print(KL_divergence(q,p)) # 0.019019266539324498

if __name__ == "__main__":
    # 样本数量可以不同，特征数目必须相同
    df = pd.read_csv('data2/ST10000NM0086.csv')
    x_train = df.drop(['ID', 'date', 'serial_number', 'model', 'capacity_bytes', 'failure'], axis=1).values
    y_train = df['failure'].values

    dataset2 = pd.read_csv('data2/ST8000NM0055.csv')
    # Extract feature values (X) from dataset
    # 删掉列
    x_test = dataset2.drop(['ID', 'date', 'serial_number','model', 'capacity_bytes','failure'], axis = 1).values
    y_test = dataset2['failure'].values

    x_train = torch.FloatTensor(x_train)
    y_train = torch.LongTensor(y_train)
    x_test = torch.FloatTensor(x_test)
    y_test = torch.LongTensor(y_test)
    # 100和90是样本数量，50是特征数目
    # data_1 = torch.tensor(np.random.normal(loc=0, scale=10, size=(100, 50)))
    # data_2 = torch.tensor(np.random.normal(loc=10, scale=10, size=(90, 50)))
    # print("MMD Loss:", mmd(data_1, data_2))

    # data_1 = torch.tensor(np.random.normal(loc=0, scale=10, size=(100, 50)))
    # data_2 = torch.tensor(np.random.normal(loc=0, scale=9, size=(80, 50)))

    #print("MMD Loss:", KL_divergence(x_train, x_test))
'''
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

class MMDLoss(nn.Module):
    '''
    计算源域数据和目标域数据的MMD距离
    Params:
    source: 源域数据（n * len(x))
    target: 目标域数据（m * len(y))
    kernel_mul:
    kernel_num: 取不同高斯核的数量
    fix_sigma: 不同高斯核的sigma值
    Return:
    loss: MMD loss
    '''
    def __init__(self, kernel_type='linear', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss
'''
if __name__ == "__main__":
    # 样本数量可以不同，特征数目必须相同
    df = pd.read_csv('data_year/ST8000NM0055.csv')
    x_train = df.drop(['ID', 'date', 'serial_number', 'model', 'capacity_bytes', 'failure'], axis=1).values
    # x_train = df[['smart_1_normalized', 'smart_3_normalized', 'smart_5_normalized',
    #                'smart_5_raw', 'smart_7_normalized', 'smart_9_normalized',
    #                'smart_187_normalized', 'smart_194_normalized',
    #                'smart_197_normalized', 'smart_197_raw']].values
    y_train = df['failure'].values
    # , 'smart_189_normalized'

    dataset2 = pd.read_csv('data_year/ST4000DM000.csv')
    # Extract feature values (X) from dataset
    # 删掉列
    x_test = dataset2.drop(['ID', 'date', 'serial_number','model', 'capacity_bytes','failure'], axis = 1).values
    # x_test = dataset2[['smart_1_normalized', 'smart_3_normalized', 'smart_5_normalized',
    #               'smart_5_raw', 'smart_7_normalized', 'smart_9_normalized',
    #               'smart_187_normalized',  'smart_194_normalized',
    #               'smart_197_normalized', 'smart_197_raw']].values
    y_test = dataset2['failure'].values

    x_train = torch.FloatTensor(x_train)
    y_train = torch.LongTensor(y_train)
    x_test = torch.FloatTensor(x_test)
    y_test = torch.LongTensor(y_test)
    
    MMD = MMDLoss()
    a = MMD(source=x_train, target=x_test)
    print(a)
'''