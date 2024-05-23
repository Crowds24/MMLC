import csv
import random

import pandas as pd


class MV:

    def __init__(self, datafile, truth_file, **kwargs):
        # change settings
        for kw in kwargs:
            setattr(self, kw, kwargs[kw])

        # initialise datafile
        self.datafile = datafile
        self.truth_file = truth_file
        t2wl, task_set, label_set = self.get_info_data()
        self.t2wl = t2wl
        self.task_set = task_set
        self.num_tasks = len(task_set)
        self.label_set = label_set
        self.num_labels = len(label_set)

    def get_info_data(self):
        if not hasattr(self, 'datafile'):
            raise BaseException('There is no datafile!')

        t2wl = {}
        task_set = set()
        label_set = set()

        f = open(self.datafile, 'r')
        reader = f.readlines()

        reader = [line.strip("\n") for line in reader]
        reader = reader[1:]

        for line in reader:
            task, worker, label = line.split(',')[0:3]
            print(task, worker, label)
            if task not in t2wl:
                t2wl[task] = {}
            t2wl[task][worker] = label

            if task not in task_set:
                task_set.add(task)

            if label not in label_set:
                label_set.add(label)


        return t2wl,task_set,label_set

    def get_accuracy(self):
        if not hasattr(self, 'truth_file'):
            raise BaseException('There is no truth file!')
        if not hasattr(self, 't2a'):
            raise BaseException('There is no aggregated answers!')

        t2truth = {}
        f = open(self.truth_file, 'r')
        reader = f.readlines()
        reader = [line.strip("\n") for line in reader]
        reader = reader[1:]

        for line in reader:
            #id, task, truth, f1, f2 = line.split(',')
            task, truth = line.split(',')[1:3]
            t2truth[task] = truth

        count = []
        for task in self.task_set:

            if self.t2a[task] == t2truth[task]:
                count.append(1)
            else:
                count.append(0)
            print(self.t2a[task]+':'+t2truth[task])
        print(count)
        j = sum(count)
        return sum(count)/len(count)

    def run(self):
        # initialization
        count = {}
        for task in self.task_set:
            count[task] = {}
            for label in self.label_set:
                count[task][label] = 0

        # compute
        for task in self.task_set:
            for worker in self.t2wl[task]:
                label = self.t2wl[task][worker]
                count[task][label] += 1
        t2a = {}
        for task in self.task_set:
            t2a[task] = min(list(self.label_set))
            for label in sorted(list(self.label_set)):
                if count[task][label] > count[task][t2a[task]]:
                     t2a[task] = label
        self.t2a = t2a
        # return self.expand(e2lpd)
        return t2a


if __name__ == '__main__':
    # datafile = '../data/Generate_data/IRT_generate_date.csv'
    # #datafile = r'../data/redundancy_2/train_data_15.csv'
    datafile = r'../data/TDG4Crowd_Text.csv'
    truth_value = r'../data/text_truth.csv'
    model = MV(datafile, truth_value)
    model.run()
    acc = model.get_accuracy()
    print(acc)

    # accs = {}
    # for b in range(450, -1, -10):
    #     datafile = r'../fillData/data/data_fill_' + str(b) + '.csv'
    #     truth_value = r'../data/text_truth.csv'
    #     model = MV(datafile, truth_value)
    #     model.run()
    #     acc = model.get_accuracy()
    #     data = pd.read_csv(datafile)
    #     label = round(data.shape[0] / 245476, 3)
    #     label = int(label * 1000)
    #
    #     if label not in accs:
    #         accs[str(label)] = acc
    # print(accs)

