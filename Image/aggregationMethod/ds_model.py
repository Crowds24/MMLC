import datetime
import math
import csv
import random
import time
from collections import Counter
from decimal import Decimal, getcontext

import numpy as np
from mpmath import mp
import matplotlib.pyplot as plt


import sys
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import label_binarize

mp.dps = 50

class DS:
    def __init__(self, datafile, **kwargs):
        e2wl, w2el, label_set = self.gete2wlandw2el(datafile)
        self.e2wl = e2wl
        self.w2el = w2el
        self.workers = self.w2el.keys()
        self.label_set = label_set
        self.initalquality = 0.7
        for kw in kwargs:
            setattr(self, kw, kwargs[kw])

    # E-step
    def Update_e2lpd(self):
        self.e2lpd = {}

        for example, worker_label_set in self.e2wl.items():
            lpd = {}
            total_weight = 0

            for tlabel, prob in self.l2pd.items():
                weight = Decimal(str(prob))
                for (w, label) in worker_label_set:
                    weight = Decimal(weight)
                    a = Decimal(str(self.w2cm[w][tlabel][label]))
                    weight *= a
                lpd[tlabel] = weight
                total_weight += weight

            for tlabel in lpd:
                if total_weight == 0:
                    # uniform distribution
                    lpd[tlabel] = 1.0 / len(self.label_set)

                else:
                    lpd[tlabel] =Decimal(str(lpd[tlabel])) / total_weight


            self.e2lpd[example] = lpd

    # M-step

    def Update_l2pd(self):

        for label in self.l2pd:
            self.l2pd[label] = 0

        for _, lpd in self.e2lpd.items():
            for label in lpd:
                self.l2pd[label] += lpd[label]

        for label in self.l2pd:
            # self.l2pd[label] *= Decimal(1.0) / len(self.e2lpd)
            self.l2pd[label] *= Decimal(str(1.0)) / len(self.e2lpd)

    def Update_w2cm(self):

        for w in self.workers:
            for tlabel in self.label_set:
                for label in self.label_set:
                    self.w2cm[w][tlabel][label] = 0

        w2lweights = {}
        for w in self.w2el:
            w2lweights[w] = {}
            for label in self.label_set:
                w2lweights[w][label] = 0
            for example, _ in self.w2el[w]:
                for label in self.label_set:
                    w2lweights[w][label] += self.e2lpd[example][label]

            for tlabel in self.label_set:
                if w2lweights[w][tlabel] == 0:
                    for label in self.label_set:
                        if tlabel == label:
                            self.w2cm[w][tlabel][label] = self.initalquality
                        else:
                            self.w2cm[w][tlabel][label] =Decimal((1 - self.initalquality) * 1.0) /Decimal((len(self.label_set) - 1))

                    continue

                for example, label in self.w2el[w]:
                    self.w2cm[w][tlabel][label] += self.e2lpd[example][tlabel] * Decimal(str(1.0)) / w2lweights[w][tlabel]

        return self.w2cm

    # initialization
    def Init_l2pd(self):
        # uniform probability distribution 均匀概率分布
        # 使用的是均匀概率分布，即假设每个标签出现的先验概率相等
        l2pd = {}
        for label in self.label_set:
            l2pd[label] = 1.0 / len(self.label_set)
        return l2pd

    # 这个函数的功能是初始化工人的混淆矩阵（confusion matrix），表示每个工人在不同的任务标签下的分类准确率。
    # w2cm[worker][tlabel][label] 表示工人 worker 在标注任务时，任务标签为 tlabel 时分类为 label 的概率。
    def Init_w2cm(self):
        w2cm = {}
        for worker in self.workers:
            w2cm[worker] = {}
            for tlabel in self.label_set:
                w2cm[worker][tlabel] = {}
                for label in self.label_set:
                    if tlabel == label:
                        w2cm[worker][tlabel][label] = self.initalquality
                    else:
                        w2cm[worker][tlabel][label] = (1 - self.initalquality) / (len(self.label_set) - 1)

        return w2cm

    def run(self, iter=10):
        self.l2pd = self.Init_l2pd()
        self.w2cm = self.Init_w2cm()

        while iter > 0:

            print(iter)
            # E-step
            self.Update_e2lpd()
            #pd.DataFrame(self.e2lpd).to_csv(f"../result/u_sim.csv")
            # M-step
            self.Update_l2pd()
            self.Update_w2cm()


            # compute the likelihood
            # print(self.computelikelihood())

            iter -= 1
        # 保存结果

        return self.e2lpd, self.w2cm
    """
         计算当前模型下所有任务标注数据的对数似然值
         e2wl: 字典，保存{task, [[worker, label],...]}
         label_set: 工人的标注种类
         
    """
    def computelikelihood(self):

        lh = 0

        for _, worker_label_set in self.e2wl.items():
            temp = 0
            for tlabel, prior in self.l2pd.items():
                inner = prior
                for worker, label in worker_label_set:
                    inner *= self.w2cm[worker][tlabel][label]
                temp += inner

            lh += math.log(temp)

        return lh

    ###################################
    # The above is the EM method (a class)
    # The following are several external functions
    ###################################

    def get_accuracy(self):
        e2truth = {}
        f = open(self.truth_file, 'r')
        reader = csv.reader(f)
        next(reader)
        # 真值
        for line in reader:
            example, truth = line[0:2]
            e2truth[example] = truth

        tcount = 0
        count = 0
        e2lpd = self.e2lpd
        #pd.DataFrame(self.e2lpd).to_csv(f"../test_iter/u_sim_ds_iter5.csv")
        # 自己加上去的
        t2a = {}
        for e in e2lpd:
            if e not in e2truth:
                continue

            temp = 0
            # 对于任务e, label表示任务e在59中上的概率。 找概率最大的作为最终的结果
            for label in e2lpd[e]:
                if temp < e2lpd[e][label]:
                    temp = e2lpd[e][label]

            candidate = []

            for label in e2lpd[e]:
                if temp == e2lpd[e][label]:
                    candidate.append(label)

            # 有多个相同的就随机选一个
            truth = random.choice(candidate)
            t2a[e] = truth

            count += 1
            # 与真值标签相同就+1
            if truth == e2truth[e]:
                tcount += 1
        return tcount * 1.0 / count

    """
            reader: 读取标注的列表[task, worker, label]
            e2wl: 字典，保存{task, [[worker, label],...]}
            w2el: 字典，保存{worker, [[task, label],...]}
            label_set: 工人的标注种类
            for: 任务如果不再e2wl中，则加入
    """
    def gete2wlandw2el(self, datafile):
        e2wl = {}
        w2el = {}
        label_set = []

        f = open(datafile, 'r')
        reader = f.readlines()
        reader = [line.strip("\n") for line in reader]

        for line in reader[1:]:

            # 下表3表示类别
            example, worker, label = line.split(',')[0:3]
            # example, worker = line[1:3]
            # 下表4表示特征1
            # label = line[4]
            # 下表4表示特征2
            # label = line[5]
            if example not in e2wl:
                e2wl[example] = []
            e2wl[example].append([worker, label])
            if worker not in w2el:
                w2el[worker] = []
            w2el[worker].append([example, label])

            if label not in label_set:
                label_set.append(label)
        return e2wl, w2el, label_set

    def get_roc(self):
        u, v = self.run()

        truth_file = r'../analyse_data/result/groundtruth.csv'

        # truth_value = pd.read_csv('../analyse_data/result/groundtruth_1000.csv').values
        truth_value = pd.read_csv(truth_file).values
        # truth_value = pd.read_csv('../simulation_data/ground_truth_4.csv').values
        # truth_file = r'../simulation_data/ground_truth_4.csv'
        # truth_value = pd.read_csv('../simulation_data/ground_truth_1000.csv').values
        # 每个任务的真值所占的概率
        n_class = len(self.label_set)
        y_gt = np.zeros(shape=(len(truth_value),), dtype=int)
        y_pred = np.zeros(shape=(len(truth_value), n_class))

        ground_truth = {}
        for i in range(len(truth_value)):
            # task, gt, gt_f1, gt_f2 = tuple(truth_value[i])
            task, gt = tuple(truth_value[i])
            ground_truth[task] = gt

            y_gt[i] = gt
            for j in range(n_class):
                y_pred[i][j] = u[str(task)][str(j)]

        y_gt_one_hot = label_binarize(y_gt, np.arange(n_class))  # 装换成类似二进制的编码

        fpr, tpr, thresholds = metrics.roc_curve(y_gt_one_hot.ravel(), y_pred.ravel())
        auc = metrics.auc(fpr, tpr)
        self.draw_roc(fpr, tpr, auc)

    def draw_roc(self, fpr, tpr, auc):
        # FPR就是横坐标,TPR就是纵坐标
        plt.plot(fpr, tpr, c='r', lw=2, alpha=0.7, label=u'AUC=%.3f' % auc)
        plt.plot((0, 1), (0, 1), c='#808080', lw=1, ls='--', alpha=0.7)
        plt.xlim((-0.01, 1.02))
        plt.ylim((-0.01, 1.02))
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.xlabel('False Positive Rate', fontsize=13)
        plt.ylabel('True Positive Rate', fontsize=13)
        plt.grid(b=True, ls=':')
        plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
        # plt.title(u'鸢尾花数据Logistic分类后的ROC和AUC', fontsize=17)
        plt.savefig("mv_roc.pdf")
        plt.show()

def test():
    u = pd.read_csv("result/u_sim_hds.csv")
    data = u.sort_values(by='0', ascending=True)
    data.to_csv('result/u1_sim.csv', index=False)

def select_17_rows(group):
    # 从每个任务的数据中随机选择17行
    return group.sample(n=17, random_state=random.randint(0, 10000))

def randamn_17():
    # 从每个任务的数据中随机选择17行
    df_17 = grouped.apply(select_17_rows)
    # 保存结果到文件
    df_17.to_csv(f'../result/data_17.csv', index=False)


if __name__ == "__main__":

    # 工人回答填充完整数据实验
    #datafile = '../data/Generate_data/IRT_generate_date.csv'
    #datafile = r'../data/data-plus.csv'
    # datafile = r'../data/filtered_df_20.csv'
    # truth_value = r'../data/result3.csv'
    #datafile = r'../data/new/train/data_train.csv'
    #truth_value = r'../data/new/train/label_train.csv'
    # datafile = r'../data/Generate_data/IRT_generate_date.csv'
    # truth_value = r'../data/new/train/label_train.csv'
    # model = DS(datafile, truth_file=truth_value)
    # model.run()
    # accuracy = model.get_accuracy()
    # print("准确率：%f" % (accuracy))


    accs = {}
    for b in range(170, -1, -5):
        datafile = r'../fillData/data/data_fill_' + str(b) + '.csv'
        truth_value = r'../data/new/train/label_train.csv'
        model = DS(datafile, truth_file=truth_value)
        model.run()
        acc = model.get_accuracy()
        data = pd.read_csv(datafile)
        label = round(data.shape[0] / 57643, 3)
        label = int(label * 1000)

        if label not in accs:
            accs[str(label)] = acc

    print(accs)

    # 随机取17个数据
    # res = []
    # df = pd.read_csv('../data/worker_answer_whole.csv')
    # grouped = df.groupby('task')
    # for i in range(0, 5):
    #     print("i:", i)
    #     randamn_17()
    #     datafile = r'../result/data_17.csv'
    #     truth_value = r'../baseLine_data/result3.csv'
    #     model = DS(datafile, truth_file=truth_value)
    #     model.run()
    #     accuracy = model.get_accuracy()
    #     print("准确率：%f" % (accuracy))
    #     res.append(accuracy)
    # pd.DataFrame(res).to_csv(f'../result/res_DS_whole.csv')
    # test()
    # datafile = r'../analyse_data/result/_data_1000_' + str(5) + '.csv'
    # truth_value = r'../analyse_data/result/groundtruth.csv'
    # model = DS(datafile, truth_file=truth_value)
    # model.run()
    # res = []
    # for i in range(5, 50,5):
    #     datafile = r'../analyse_data/result/w_data_1000_' + str(i) + '.csv'
    #     truth_file = r'../analyse_data/result/groundtruth_1000.csv'
    #
    #     model = DS(datafile, truth_file=truth_file)
    #     model.run()
    #     accuracy = model.get_accuracy()
    #     print("准确率：%f" % (accuracy))
    #     res.append(accuracy)
    # pd.DataFrame(res).to_csv('result/w_sim_res_DS.csv')
    # 统计时间
    # start = time.time()
    # datafile = r'../analyse_data/result/answer7_' + str(10) + '.csv'
    # truth_value = r'../analyse_data/result/groundtruth.csv'
    # model = DS(datafile, truth_file=truth_value)
    # model.get_roc()
    # print(model.get_accuracy())
    # end = time.time()
    # print(end-start)

    #统计类别和特征的准确率
    # datafile = r'../analyse_data/result/f1_f2.csv'
    # truth_file = r'../analyse_data/result/result3.csv'
    #
    # model = DS(datafile, truth_file=truth_file)
    # model.run()
    # accuracy = model.get_accuracy()
    # print("准确率：%f" % (accuracy))

    # 统计时间
    # start = datetime.datetime.now()
    # # 这里可以放入运行代码，比较直接
    # datafile = r'../analyse_data/result/answer7_' + str(17) + '.csv'
    # truth_file = r'../analyse_data/result/result3.csv'
    #
    # model = DS(datafile, truth_file=truth_file)
    # model.run()
    # accuracy = model.get_accuracy()
    # print("准确率：%f" % (accuracy))
    # end = datetime.datetime.now()
    #
    # print(end - start)


    # datafile = r'../analyse_data/result/answer7_' + str(10) + '.csv'
    # truth_file = r'../analyse_data/result/groundtruth.csv'
    #
    # model = DS(datafile, truth_file=truth_file)
    # model.run()
    # accuracy,t2a = model.get_accuracy()
    # print("准确率：%f" % (accuracy))
    # ds = Counter(t2a.values()).items()
    # ds = dict(ds)
    #
    #
    # df = pd.read_csv('../analyse_data/result/groundtruth.csv').values
    # li = Counter(df[:, 1]).items()
    # li = sorted(li, key=lambda x: x[1], reverse=True)
    # x1 = []
    # y1 = []
    # for i, j in li:
    #     x1.append(i)
    #
    # for x in x1:
    #     if str(x) in ds.keys():
    #         y1.append(np.log(ds[str(x)] + 1))
    #     else:
    #         y1.append(0)
    #
    # plt.plot(sorted(x1), y1)
    # plt.xlabel("DS")
    # plt.ylabel("log(N+1)")
    # plt.show()



