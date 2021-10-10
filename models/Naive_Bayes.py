import numpy as np
from collections import Counter
from tqdm import tqdm
from DataSet import Doc2Vec
from multiprocessing import Process, Manager
from sklearn.metrics import precision_recall_fscore_support

class Naive_Bayes():
    def __init__(self):
        pass

    def train(self, TrainMat, Label):
        """
        :param TrainMat:[M, D]
        :param Label:[M,1]
        :return: P_mat:[n_class,D], P_class:[n_class,1]
        """
        M, D = TrainMat.shape
        n_class = len(set(list(Label)))
        P_class = np.zeros(n_class)
        for i in range(n_class):
            idx = list(np.where(Label == i))[0]
            P_class[i] = len(idx) / M

        P_mat = np.zeros((n_class, D))
        for i in range(n_class):
            idx = list(np.where(Label == i))[0]
            freq_vec = TrainMat[idx].sum(axis=0)   # 该类的词汇频次求和
            freq_vec += 1                  # Laplace smooth
            num_word = np.sum(freq_vec)
            P_mat[i] = np.log(freq_vec / num_word)
        return P_mat, P_class

    def test1(self, Test_DocList, gt_label, VocabList, P_mat, P_class, process, result):

        M = len(gt_label)
        pred_label = np.zeros(M)
        for i in tqdm(range(M)):
            Doc = Test_DocList[i]
            vec = Doc2Vec(VocabList, Doc).reshape(1, len(VocabList))
            logp = np.sum(vec * P_mat, axis=1)
            logp = logp + P_class
            pred_label[i] = np.argmax(logp)
        diff = pred_label - gt_label
        idx = list(np.where(diff == 0))[0]
        accuracy = len(idx) / M
        result[process] = [accuracy, M]


    def test(self, X_test, y_gt,  P_mat, P_class):

        M = len(y_gt)
        pred_label = np.zeros(M)
        for i in range(M):
            vec = X_test[i]
            logp = np.sum(vec * P_mat, axis=1)
            logp = logp + P_class
            pred_label[i] = np.argmax(logp)
        precision, recall, _, _ = precision_recall_fscore_support(y_gt,
                                                                  pred_label)
        diff = pred_label - y_gt
        idx = list(np.where(diff == 0))[0]
        accuracy = len(idx) / M
        return accuracy, recall


    def MultiProcessTest(self, Test_DocList, gt_label, VocabList, P_mat, P_class):
        num_pro = 8
        M = int(len(gt_label) / num_pro)
        result = Manager().dict()  # 返回值
        jobs = []  # 管理线程
        for i in range(num_pro):
            if i < num_pro - 1:
                s, e = i*M, i*M+M
            else:
                s, e = i*M, -1
            p = Process(target=self.test1, args=(Test_DocList[s:e], gt_label[s:e],
                                                  VocabList, P_mat, P_class, i, result))
            jobs.append(p)
            p.start()

        for pro in jobs:
            pro.join()

        acc = np.array([v[0] for _, v in result.items()])
        num = np.array([v[1] for _, v in result.items()])
        accuracy = np.sum((acc * num)) / len(gt_label)

        return accuracy
