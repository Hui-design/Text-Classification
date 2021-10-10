import numpy as np
import torch
from models.Naive_Bayes import Naive_Bayes
from models.Knn import Knn
from models.MLPs import MLPs, Net
from DataSet import loadDataSet, createVocabList, Doc2Mat, find_common_highFreq, find_low_freq
from sklearn.model_selection import KFold
from AverageMeter import AverageMeter
from collections import Counter
from tqdm import tqdm
import os
import torch.optim as optim
import logging
import sys

if __name__ == '__main__':
    Test_NB = False
    Test_Knn = True
    Test_MLPs = False

    # 预处理数据
    DocList, ClassVec = loadDataSet()
    high_freq = find_common_highFreq(DocList, ClassVec)   # 共有的高频词没有意义
    low_freq = find_low_freq(DocList, ClassVec)    # 每个类只出现个数次的是拼写错误
    VocabList = createVocabList(DocList)
    VocabList = list(set(VocabList) - set(high_freq) - set(low_freq))
    DataMat = Doc2Mat(VocabList, DocList)
    np.save('Data/DataMat.npy', DataMat)
    np.save('Data/ClassVec.npy', ClassVec)
    np.save('Data/VocabList.npy', VocabList)
    # X = np.load('Data/DataMat.npy', allow_pickle=True)
    # y = np.load('Data/ClassVec.npy', allow_pickle=True)
    # VocabList = list(np.load('Data/VocabList.npy', allow_pickle=True))

    # 把X顺序先打乱
    np.random.seed(3)
    M = X.shape[0]
    idx = np.random.choice(M, M, replace=False)
    X, y = X[idx], y[idx]

    # 交叉验证
    num_cross = 5
    kf = KFold(n_splits=num_cross, random_state=None)
    Accuracy = AverageMeter()
    recall = np.zeros(20)
    if not Test_MLPs:
        for train_index, test_index in tqdm(kf.split(X)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            if Test_NB:
                model = Naive_Bayes()
                P_mat, P_class = model.train(X_train, y_train)
                Acc, recall_i = model.test(X_test, y_test, P_mat, P_class)
                Accuracy.update(Acc)
                recall += recall_i
            if Test_Knn:
                k = 15
                model = Knn(k)
                train_mat, train_class = model.train(X_train, y_train)
                Acc, recall_i = model.test(X_test, y_test, train_mat, train_class)
                Accuracy.update(Acc)
                recall += recall_i
        recall /= 5
        print(f"R_min:{np.min(recall)} {np.argmin(recall)}\t"
              f"R_max:{np.max(recall)} {np.argmax(recall)}\t"
              f"R_mean:{np.mean(recall)} {np.std(recall)}")
        print(f"Accuracy:{Accuracy.avg * 100:.2f}%")

    if Test_MLPs:
        model = MLPs()
        X_train, X_test = X[:int(0.8*M)], X[int(0.8*M):]
        y_train, y_test = y[:int(0.8*M)], y[int(0.8*M):]
        model.train(X_train,y_train)
        # state = torch.load('snapshot/MLPs_10_08_3.pth')
        weights = state['state_dict']
        net = Net()
        net.load_state_dict(weights)
        Accuracy, recall = model.test(net.cuda(), X_test, y_test)
        print(f"R_min:{np.min(recall)} {np.argmin(recall)}\t"
              f"R_max:{np.max(recall)} {np.argmax(recall)}\t"
              f"R_mean:{np.mean(recall)} {np.std(recall)}")
        print(f"Accuracy:{Accuracy * 100:.2f}%")
