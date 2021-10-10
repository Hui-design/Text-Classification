import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging,sys


# 配置log文件
log_filename = f'snapshot/output.log'
logging.basicConfig(level=logging.INFO, filename=log_filename, filemode='a')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.classification = nn.Sequential(
            nn.Conv1d(15306, 256, kernel_size=1, bias=True),
            # nn.MaxPool1d((500,1), stride=1),
            nn.Sigmoid(),
            # # nn.Conv1d(128, 256, kernel_size=1, bias=True),
            # # nn.ReLU(inplace=True),
            # # nn.Conv1d(256, 128, kernel_size=1, bias=True),
            # # nn.ReLU(inplace=True),
            # # nn.Conv1d(128, 32, kernel_size=1, bias=True),
            # # nn.ReLU(inplace=True),
            nn.Conv1d(256, 20, kernel_size=1, bias=True),
            nn.Softmax(dim=1)  # nn.Sigmoid()
        )

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = self.classification(x)
        # x = nn.Conv1d(11599, 20, kernel_size=1, bias=True)(x)
        # x = nn.Softmax(dim=1)(x)
        return x


class MLPs():
    def __init__(self):
        self.net = Net()
        self.optimizer = optim.SGD(self.net.parameters(), lr=5)
        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=0.98,
        )


    def train(self, X_train, gt_label):

        self.net.train()
        self.net.cuda()
        X_mean = np.mean(X_train, axis=1, keepdims=True)
        X_std = np.std(X_train, axis=1, keepdims=True)
        X_train = (X_train - X_mean) / X_std
        input = torch.from_numpy(X_train)[None, :, :].permute(0, 2, 1).to(torch.float32)
        input = input.to('cuda')
        for epoch in tqdm(range(1000)):
            # 归一化
            # in your training loop:
            self.optimizer.zero_grad()  # zero the gradient buffers
            pred = self.net(input).squeeze(0).permute(1, 0)
            gt = torch.from_numpy(gt_label).view(-1).long()
            gt = gt.to('cuda')
            criterion = nn.CrossEntropyLoss()
            loss = criterion(pred, gt)
            pred = torch.max(pred, 1)[1]
            pred = pred.squeeze()
            precision, recall, _, _ = precision_recall_fscore_support(gt.detach().cpu().numpy(), pred.detach().cpu().numpy())
            loss.backward()  # 计算梯度
            self.optimizer.step()  # 更新参数
            if epoch % 20 == 0:
                if self.scheduler.get_last_lr()[0] > 0.001:
                    self.scheduler.step()
                logging.info(f"epoch:{epoch}\t"
                    f"loss: {loss.item():.4f}\t"
                      f"recall: {recall.mean():.4f}\t"
                      f"precision:{precision.mean():.4f}")
        stat = {'state_dict': self.net.state_dict()}
        torch.save(stat, 'snapshot/MLPs_10_08_3.pth')

    def test(self, net, X_test, gt_label):
        self.net = net
        # self.net.cuda()
        with torch.no_grad():
            # 归一化
            X_mean = np.mean(X_test, axis=1, keepdims=True)
            X_std = np.std(X_test, axis=1, keepdims=True)
            X_test = (X_test - X_mean) / X_std
            input = torch.from_numpy(X_test[None, :, :]).permute(0, 2, 1).to(torch.float32)
            input = input.to('cuda')
            pred = self.net(input).squeeze(0).permute(1, 0)
            gt = torch.from_numpy(gt_label).view(-1).long()
            gt = gt.to('cuda')
            criterion = nn.CrossEntropyLoss()
            loss = criterion(pred, gt)
            pred = torch.max(pred, 1)[1]
            pred = pred.squeeze()
            diff = pred.detach().cpu().numpy() - gt.detach().cpu().numpy()
            precision, recall, _, _ = precision_recall_fscore_support(gt.detach().cpu().numpy(),
                                                                      pred.detach().cpu().numpy())
            idx = list(np.where(diff == 0))[0]
            accuracy = len(idx) / len(diff)
            return accuracy, recall





