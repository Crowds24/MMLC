import copy
import datetime
import decimal
import math
import csv
import random
import time
from decimal import Decimal

import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import label_binarize
import time

class HDS:
    def __init__(self, datafile, truth_file, **kwargs):
        self.datafile = datafile
        self.truth_file = truth_file
        e2wl, w2el, label_set = self.gete2wlandw2el()
        self.e2wl = e2wl   # {t0:[w0:l0], t1:[w1:l1], ...}
        self.w2el = w2el    # {w0:[t0:l0], w1:[t1:l1], ...}
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
                weight = prob
                for (w, label) in worker_label_set:
                    weight = Decimal(str(weight))
                    a = Decimal(str(self.w2cm[w][tlabel][label]))
                    weight *= a

                lpd[tlabel] = weight
                total_weight += weight
            for tlabel in lpd:
                if total_weight == 0:
                    # uniform distribution
                    lpd[tlabel] = 1.0 / len(self.label_set)
                else:
                    lpd[tlabel] = Decimal(str(lpd[tlabel]))/ total_weight

            self.e2lpd[example] = lpd

        # print(self.e2lpd)  # 推断
    # M-step

    def Update_l2pd(self):
        for label in self.l2pd:
            self.l2pd[label] = 0
        for _, lpd in self.e2lpd.items():
            for label in lpd:
                self.l2pd[label] += lpd[label]

        for label in self.l2pd:
            self.l2pd[label] *= Decimal('1.0') / len(self.e2lpd)
        # print(self.l2pd)  # 更新先验
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
                            self.w2cm[w][tlabel][label] = (1 - self.initalquality) * 1.0 / (len(self.label_set) - 1)

                    continue

                for example, label in self.w2el[w]:
                    self.w2cm[w][tlabel][label] += self.e2lpd[example][tlabel] * Decimal('1.0') / w2lweights[w][tlabel]


        for w in self.workers:
            for tlabel in self.label_set:
                a = 0
                for label in self.label_set:
                    if tlabel == label:
                        a = Decimal(str((1 - self.w2cm[w][tlabel][label]))) * Decimal('1.0') / Decimal(str((len(self.label_set) - 1)))
                for label in self.label_set:
                    if tlabel != label:
                        self.w2cm[w][tlabel][label] = a

        return self.w2cm

    # initialization
    def Init_l2pd(self):
        # uniform probability distribution
        l2pd = {}
        for label in self.label_set:
            l2pd[label] = 1.0 / len(self.label_set)
        return l2pd

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

    def run(self, iter=20, threshold=1e-4):

        self.l2pd = self.Init_l2pd()   # {l0:p0, l1:p1, l2:p2, l3:p3}
        self.w2cm = self.Init_w2cm()
                                        # {'78': {'1': {'1': 0.7, '3': 0.1, '4': 0.1, '2': 0.1},
                                        #         '3': {'1': 0.1, '3': 0.7, '4': 0.1, '2': 0.1},
                                        #         '4': {'1': 0.1, '3': 0.1, '4': 0.7, '2': 0.1},
                                        #         '2': {'1': 0.1, '3': 0.1, '4': 0.1, '2': 0.7}
                                        #         }
                                        #  '2': {'1': {'1': 0.7, '3': 0.1, '4': 0.1, '2': 0.1},
                                        #        '3': {'1': 0.1, '3': 0.7, '4': 0.1, '2': 0.1},
                                        #        '4': {'1': 0.1, '3': 0.1, '4': 0.7, '2': 0.1},
                                        #        '2': {'1': 0.1, '3': 0.1, '4': 0.1, '2': 0.7}
                                        #        }
                                        #  }

        Q = self.computelikelihood()
        while iter > 0:
            lastQ = Q
            print(iter)
            # E-step
            self.Update_e2lpd()
            # M-step
            self.Update_l2pd()
            self.Update_w2cm()
            # compute the likelihood
            # print self.computelikelihood()
            Q = self.computelikelihood()
            if (math.fabs(float((Q-lastQ)/lastQ))) < threshold:
                break
            iter -= 1
        # 保存结果
        #pd.DataFrame(self.e2lpd).to_csv(f"../result/u_sim.csv")
        return self.e2lpd, self.w2cm

    def computelikelihood(self):

        lh = 0

        for _, worker_label_set in self.e2wl.items():
            temp = Decimal(str('0.0'))
            for tlabel, prior in self.l2pd.items():
                inner = prior
                for worker, label in worker_label_set:
                    inner = Decimal(str(inner))
                    inner *= Decimal(str(self.w2cm[worker][tlabel][label]))
                temp += inner
            if temp <= Decimal(str('0.0')):
                lh += Decimal(str(float('-inf')))
            else:
                lh += Decimal.ln(temp)

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

        for line in reader:
            example, truth = line[1:3]
            # example, truth = line[1:3]
            # example = line[1]
            # truth = line[4]

            e2truth[example] = truth

        tcount = 0
        count = 0
        e2lpd = self.e2lpd
        for e in e2lpd:
            if e not in e2truth:
                continue
            temp = 0
            for label in e2lpd[e]:
                if temp < e2lpd[e][label]:
                    temp = e2lpd[e][label]
            candidate = []
            for label in e2lpd[e]:
                if temp == e2lpd[e][label]:
                    candidate.append(label)
            truth = random.choice(candidate)
            count += 1
            if truth == e2truth[e]:
                tcount += 1

        return tcount * 1.0 / count

    def gete2wlandw2el(self):
        e2wl = {}
        w2el = {}
        label_set = []

        f = open(datafile, 'r')
        reader = f.readlines()
        reader = [line.strip("\n") for line in reader]

        for line in reader:
            # example, worker, label = line[1:4]
            example, worker, label = line.split(',')[0:3]
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

        truth_file = r'../baseLine_data/result3.csv'

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
        plt.savefig("hds_roc.pdf")
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

if __name__ == '__main__':
    # datafile = '../data/Generate_data/HDS_generate_date.csv'
    # #datafile = r'../data/redundancy_2/train_data_3.csv'
    # # datafile = r'../data/filtered_df_20.csv'
    datafile = r'../data/TDG4Crowd_Text.csv'
    truth_value = r'../data/text_truth.csv'
    model = HDS(datafile, truth_file=truth_value)
    model.run()
    accuracy = model.get_accuracy()
    print("准确率：%f" % (accuracy))

    # accs = {}
    # for b in range(450, -1, -10):
    #     datafile = r'../fillData/data/data_fill_' + str(b) + '.csv'
    #     truth_value = r'../data/text_truth.csv'
    #     model = HDS(datafile, truth_file=truth_value)
    #     model.run()
    #     acc = model.get_accuracy()
    #     data = pd.read_csv(datafile)
    #     label = round(data.shape[0] / 245476, 3)
    #     label = int(label * 1000)
    #
    #     if label not in accs:
    #         accs[str(label)] = acc
    # print(accs)

