from sklearn.neighbors import KNeighborsClassifier
import sys
sys.path.append("..")
from collections import Counter
from DataSet import Doc2Mat
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

class Knn():
    def __init__(self, K):
        self.K = K
        self.model = KNeighborsClassifier(n_neighbors=K, metric='euclidean')

    def train(self, Train_mat, Train_ClassVec):
        self.model.fit(Train_mat, Train_ClassVec) # 训练
        return Train_mat, Train_ClassVec

    def vote(self, indices, train_label):
        len = indices.shape[0]
        pred = np.zeros(len)
        for i in range(len):
            idx = indices[i].squeeze()
            label_i = train_label[idx]
            if isinstance(label_i, np.int32):  # 处理k=1的特殊情况
                pred[i] = label_i
            else:
                freq = Counter(label_i)
                pred[i] = max(freq, key=freq.get)
        return pred

    def test(self, Test_mat, gt_label, train_mat, train_label):
        # pred_label = self.model.predict(Test_mat) # 预测
        _, indices = self.model.kneighbors(Test_mat)  # [M, K]
        pred_label = self.vote(indices, train_label)
        precision, recall, _, _ = precision_recall_fscore_support(gt_label,
                                                                  pred_label)
        diff = pred_label - gt_label
        idx = list(np.where(diff == 0))[0]
        accuracy = len(idx) / len(gt_label)
        return accuracy, recall