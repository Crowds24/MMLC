import csv
import random

import numpy as np

from Untils.sample_data import sampling


class MV_generate:

    def __init__(self, crowd_file, truth_file, **kwargs):

        # initialise datafile
        self.crowd_file = crowd_file
        self.truth_file = truth_file
        t2wl, task_set, label_set, worker_set = self.get_info_data()
        self.t2wl = t2wl
        self.task_set = task_set
        self.worker_set = worker_set
        self.num_tasks = len(task_set)
        self.label_set = label_set
        self.num_labels = len(label_set)

    def get_info_data(self):
        t2wl = {}
        task_set = set()
        worker_set = set()
        label_set = set()

        # f = open(self.datafile, 'r')
        # reader = csv.reader(f)
        # next(reader)

        f = open(self.crowd_file, 'r')
        reader = f.readlines()
        reader = [line.strip("\n") for line in reader]

        for line in reader[1:]:
            task, worker, label = line.split(',')[0:3]
            print(task, worker, label)
            if task not in t2wl:
                t2wl[task] = {}

            t2wl[task][worker] = label  # {t0:{w0：l0, w1:l1, ... }}

            if task not in task_set:
                task_set.add(task)

            if label not in label_set:
                label_set.add(label)

            if worker not in worker_set:
                worker_set.add(worker)


        return t2wl, task_set, label_set, worker_set

    def get_accuracy(self):
        if not hasattr(self, 'truth_file'):
            raise BaseException('There is no truth file!')
        if not hasattr(self, 't2a'):
            raise BaseException('There is no aggregated answers!')

        count = []

        # f = open(self.truth_file, 'r')
        # reader = csv.reader(f)
        # next(reader)

        f = open(self.truth_file, 'r')
        reader = f.readlines()
        reader = [line.strip("\n") for line in reader]

        for line in reader:
            task, truth = line.split('\t')
            if task not in self.t2a:
                print('task %s in not found in the list of answers' % task)
            elif self.t2a[task] == truth:
                count.append(1)
            else:
                count.append(0)

        return sum(count) / len(count)
    '''
    t2p表示820个任务工人标注某个类别的概率
    '''
    def run(self):
        # initialization
        count = {}  # {t:{l0:0, l1:0 , l2:0, ...}}
        for task in self.task_set:
            count[task] = {}
            for label in self.label_set:
                count[task][label] = 0.0

        # compute
        for task in self.task_set:
            for worker in self.t2wl[task]:
                label = self.t2wl[task][worker]
                count[task][label] += 1.0


        # compute_p
        t2p = count.copy()  # {t:{l0:p0, l1:p1 , l2:p2, ...}}

        for task in self.task_set:
            sum = len(self.t2wl[task].keys())
            for label in self.label_set:
                t2p[task][label] = t2p[task][label]/sum
        print('t2p', t2p)
        self.t2p = t2p


    def generate(self, sample_file, generate_file, test_loader):
        # generate
        e2truth = {}
        f = open(self.truth_file, 'r')
        reader = f.readlines()
        reader = [line.strip("\n") for line in reader]

        for line in reader[1:]:
            example, truth = line.split(',')[1:3]
            e2truth[example] = truth
        f.close()
        f_open = open(sample_file, 'r')
        reader = f_open.readlines()
        reader = [line.strip("\n") for line in reader]
        f_save = open(generate_file, 'w')
        for line in reader[1:]:

            task, worker, label = line.split(',')[1:4]
            l2p = {}
            for label in self.label_set:
                if label == e2truth[task]:
                    l2p[label] = self.t2p[task][e2truth[task]]
                else:
                    l2p[label] = (1 - self.t2p[task][e2truth[task]])/(len(self.label_set) - 1)
            p = np.array(list(l2p.values()))
            label = np.random.choice(list(l2p.keys()), p=p.ravel())
            f_save.write(task+'\t'+worker+'\t'+label+'\n')

        f_open.close()
        f_save.close()


    def generate_fixed_annotator(self, exist_annotator, generate_file, test_loader):
        # generate
        e2truth = {}
        f = open(self.truth_file, 'r')
        reader = f.readlines()
        reader = [line.strip("\n") for line in reader]

        for line in reader:
            example, truth = line.split('\t')
            e2truth[example] = truth
        f.close()



        f_open = open(self.crowd_file, 'r')
        reader = f_open.readlines()
        reader = [line.strip("\n") for line in reader]
        f_save = open(generate_file, 'w')
        for line in reader:

            task, worker, label = line.split('\t')
            if worker in exist_annotator:
                f_save.write(task + '\t' + worker + '\t' + label + '\n')
            else:
                l2p = {}
                for label in self.label_set:
                    if label == e2truth[task]:
                        l2p[label] = self.t2p[task][e2truth[task]]
                    else:
                        l2p[label] = (1 - self.t2p[task][e2truth[task]])/(len(self.label_set) - 1)
                p = np.array(list(l2p.values()))
                label = np.random.choice(list(l2p.keys()), p=p.ravel())
                f_save.write(task+'\t'+worker+'\t'+label+'\n')

        f_open.close()
        f_save.close()

    def generate_fixed_task(self, exist_task, generate_file, test_loader):
        # generate
        e2truth = {}
        f = open(self.truth_file, 'r')
        reader = f.readlines()
        reader = [line.strip("\n") for line in reader]

        for line in reader:
            example, truth = line.split('\t')
            e2truth[example] = truth
        f.close()


        f_open = open(self.crowd_file, 'r')
        reader = f_open.readlines()
        reader = [line.strip("\n") for line in reader]
        f_save = open(generate_file, 'w')
        for line in reader:
            task, worker, label = line.split('\t')
            if task in exist_task:
                f_save.write(task + '\t' + worker + '\t' + label + '\n')
            else:
                l2p = {}
                for label_item in self.label_set:
                    if label_item == e2truth[task]:
                        l2p[label_item] = self.t2p[task][e2truth[task]]
                    else:
                        l2p[label_item] = (1 - self.t2p[task][e2truth[task]])/(len(self.label_set) - 1)
                p = np.array(list(l2p.values()))
                new_label = np.random.choice(list(l2p.keys()), p=p.ravel())
                f_save.write(task+'\t'+worker+'\t'+new_label+'\n')

        f_open.close()
        f_save.close()

    def generate_replenish(self, exist_task, generate_file, test_loader):
        e2truth = {}
        f = open(self.truth_file, 'r')
        reader = f.readlines()
        reader = [line.strip("\n") for line in reader]

        for line in reader[1:]:

            example, truth = line.split(',')[0:2]
            print(example, truth)
            e2truth[example] = truth
        f.close()


        f_open = open(self.crowd_file, 'r')
        reader = f_open.readlines()
        reader = [line.strip("\n") for line in reader]

        worker_id_list = []
        example_id_list = []
        for line in reader[1:]:
            example, worker, label = line.split(',')[0:3]
            if worker not in worker_id_list:
                worker_id_list.append(worker)
            if example not in example_id_list:
                example_id_list.append(example)

        e2wl = {}
        for line in reader[1:]:
            example, worker, label = line.split(',')[0:3]

            if example not in e2wl:
                e2wl[example] = {}
                for worker_id in worker_id_list:
                    e2wl[example][worker_id] = -1

            e2wl[example][worker] = label

        f_save = open(generate_file, 'w')
        for example, w2l in e2wl.items():
            for worker, label in w2l.items():
                if example in exist_task:
                    if label != -1:
                        f_save.write(example + '\t' + worker + '\t' + label + '\n')
                else:
                    if label == -1:
                        p = np.array(list(self.t2p[example].values()))
                        new_label = np.random.choice(list(self.t2p[example].keys()), p=p.ravel())
                        f_save.write(example + ',' + worker + ',' + new_label + '\n')

                    else:
                        f_save.write(example + ',' + worker + ',' + label + '\n')
        f_open.close()
        f_save.close()

if __name__ == '__main__':
    crowd_file = r'../data/music_feature.csv'
    truth_file = r'../data/music_true.csv'
    exist_task = sampling(crowd_file, 0).run_fixed_task()
    mv = MV_generate(crowd_file, truth_file)
    generate_file = '../data/Generate_data/MV_generate_date.csv'
    test_loader = None
    mv.run()
    mv.generate_replenish(exist_task, generate_file, None)