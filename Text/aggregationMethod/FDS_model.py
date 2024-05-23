from __future__ import print_function

import csv

import numpy as np
import pandas as pd
import sys


def FDS(datafile, goldfile):
    """
    Run the EM estimator on the data passed as the parameter
    Args:
        datafile: a dictionary object of crowed-sourced responses:{questions: {participants: [labels]}}
        goldfile: The correct label for each question: [nQuestions]
    Returns:
        result: The estimated label for each question: [nQuestions]
        acc: Accuracy of the estimated labels if gold was specified
    """
    data = pd.read_csv(datafile, index_col=None, header=0)
    print(data)
    data = data.iloc[:, :3]
    data = data.rename(columns=dict(zip(data.columns, ['taskId', 'worker', 'label'])))
    data = data.sort_values('taskId')
    print(data)
    result_dict = {}

    # 遍历 DataFrame 的每一行
    for _, row in data.iterrows():
        task_id = row["taskId"]
        worker_id = row["worker"]
        label = row["label"]

        # 如果任务编号不存在于字典中，则在字典中添加该任务编号并初始化为一个空字典
        if task_id not in result_dict:
            result_dict[task_id] = {}

        if worker_id in result_dict[task_id]:
            # 如果工人编号已经存在，则将标注添加到现有的列表中
            result_dict[task_id][worker_id].append(label)
        else:
            # 如果工人编号不存在，则创建一个新的列表并添加标注
            result_dict[task_id][worker_id] = [label]
    keys = list(result_dict.keys())

    result = run(result_dict)
    # 真值
    gold = pd.read_csv(truth_value, index_col=0, header=0)
    gold = gold.iloc[:, :2]
    print(gold)
    gold = gold.rename(columns=dict(zip(gold.columns, ['taskId', 'label'])))
    print(gold)

    # 创建一个字典，以任务为键，标签为值
    task_label_dict = gold.set_index('taskId')['label'].to_dict()

    # 根据键值列表获取对应的标签
    gold = [task_label_dict.get(key) for key in keys]


    if goldfile is not None:
        acc = (gold == result).mean()
    else:
        acc = None
    print(acc)
    return result, acc


def run(responses, tol=0.0001, max_iter=100):
    """
    Run the aggregator on response data
    Args:
        responses: a dictionary object of responses: {questions: {participants: [labels]}}
        tol: threshold for class marginals for convergence of the algorithm
        CM_tol: threshold for class marginals for switching to 'hard' mode
            in Hybrid algorithm. Has no effect for FDS or DS
        max_iter: maximum number of iterations of EM
    Returns:
        The estimated label for each question: [nQuestions]
    """

    # convert responses to counts
    # 分别获取问题、参与者、类别的集合 以及counts
    (questions, participants, classes, counts) = responses_to_counts(responses)

    # 初始化维度与counts相同的矩阵
    question_classes = initialize(counts)

    # initialize
    nIter = 0
    converged = False
    old_class_marginals = None

    while not converged:
        nIter += 1
        print(nIter)
        # M-step
        (class_marginals, error_rates) = m_step(counts, question_classes)

        # E-step
        question_classes = e_step(counts, class_marginals, error_rates)

        # check for convergence
        if old_class_marginals is not None:
            class_marginals_diff = np.sum(np.abs(class_marginals - old_class_marginals))

            if (class_marginals_diff < tol) or nIter >= max_iter:
                converged = True

        old_class_marginals = class_marginals

    np.set_printoptions(precision=2, suppress=True)

    result = np.argmax(question_classes, axis=1)

    return result


def responses_to_counts(responses):
    """
    Convert a matrix of annotations to count data
    Args:
        responses: dictionary of responses {questions:{participants:[responses]}}
    Returns:
        questions: list of questions
        participants: list of participants
        classes: list of possible classes (choices)
        counts: 3d array of counts: [questions x participants x classes]
    """
    questions = responses.keys()
    questions = sorted(questions)
    nQuestions = len(questions)

    # determine the participants and classes 确定参与者和类别
    participants = set()
    classes = set()
    for i in questions:
        # 参与i任务标注的工人
        i_participants = responses[i].keys()
        for k in i_participants:
            if k not in participants:
                participants.add(k)
            ik_responses = responses[i][k]
            classes.update(ik_responses)

    classes = list(classes)
    classes.sort()
    nClasses = len(classes)

    participants = list(participants)
    participants.sort()
    nParticipants = len(participants)

    # create a 3d array to hold counts
    counts = np.zeros([nQuestions, nParticipants, nClasses])

    # convert responses to counts
    for question in questions:
        i = questions.index(question)
        for participant in responses[question].keys():
            k = participants.index(participant)
            for response in responses[question][participant]:
                j = classes.index(response)
                counts[i, k, j] += 1

    return (questions, participants, classes, counts)


def initialize(counts):
    """
    Get majority voting estimates for the true classes using counts
    Args:
        counts: counts of the number of times each response was received by each question from each participant: [questions x participants x classes]
    Returns:
        question_classes: matrix                                                                                                                                   of estimates of true classes:
            [questions x responses]
    --------------------------------------------------------------
    使用计数获得真实类别的多数投票估计
    Args:
        counts: 每个参与者每个问题收到每个回答的次数[questions x participants x classes]
    Return:
        question_classes: matrix[questions x responses]
    """

    [nQuestions, nParticipants, nClasses] = np.shape(counts)
    response_sums = np.sum(counts, 1)
    question_classes = np.zeros([nQuestions, nClasses])

    for p in range(nQuestions):
        indices = np.argwhere(response_sums[p, :] == np.max(response_sums[p, :])).flatten()
        question_classes[p, np.random.choice(indices)] = 1

    return question_classes


def m_step(counts, question_classes):
    """
    M Step for the EM algorithm
    Get estimates for the prior class probabilities (p_j) and the error
    rates (pi_jkl) using MLE with current estimates of true question classes
    See equations 2.3 and 2.4 in Dawid-Skene (1979) or equations 3 and 4 in
    our paper (Fast Dawid-Skene: A Fast Vote Aggregation Scheme for Sentiment
    Classification)
    Args:
        counts: Array of how many times each response was received by each question from each participant: [questions x participants x classes]
        question_classes: Matrix of current assignments of questions to classes
    Returns:
        p_j: class marginals - the probability that the correct answer of a question is a given choice (class) [classes]
        pi_kjl: error rates - the probability of participant k labeling response l for a question whose correct answer is j [participants, classes, classes]
    -------------------------------------------------------------------------------------------------
    Args:
        counts: 每个参与者的每个问题收到每个回答的次数数组
        question_classes:当前给类别分配问题的矩阵
    Returns:
        p_j: class marginals 问题的正确答案是给定选择（类）的概率
        pi_kjl: error rates 对于正确答案为j的问题，参与者k标记响应l的概率
    """

    [nQuestions, nParticipants, nClasses] = np.shape(counts)

    # compute class marginals
    class_marginals = np.sum(question_classes, 0) / float(nQuestions)

    # compute error rates
    error_rates = np.zeros([nParticipants, nClasses, nClasses])
    for k in range(nParticipants):
        for j in range(nClasses):
            for l in range(nClasses):
                error_rates[k, j, l] = np.dot(question_classes[:, j], counts[:, k, l])
            sum_over_responses = np.sum(error_rates[k, j, :])
            if sum_over_responses > 0:
                error_rates[k, j, :] = error_rates[k, j, :] / float(sum_over_responses)

    return (class_marginals, error_rates)


def e_step(counts, class_marginals, error_rates):
    """
    E (+ C) Step for the EM algorithm
    Determine the probability of each question belonging to each class,
    given current ML estimates of the parameters from the M-step. Also
    perform the C step (along with E step (see section 3.4)) in case of FDS.
    See equation 2.5 in Dawid-Skene (1979) or equations 1 and 2 in
    our paper (Fast Dawid Skene: A Fast Vote Aggregation Scheme for Sentiment
    Classification)
    Args:
        counts: Array of how many times each response was received
            by each question from each participant: [questions x participants x classes]
        class_marginals: probability of a random question belonging to each class: [classes]
        error_rates: probability of participant k assigning a question whose correct
            label is j the label l: [participants x classes x classes]
        mode: One among ['H', 'Hphase2', 'FDS', 'DS']
            'Hphase2' and 'FDS' will perform E + C step
            'DS' and 'H' will perform only the E step
            'FDS': use for FDS algorithm
            'DS': use for original DS algorithm
            'H' and 'Hphase2': use for Hybrid algorithm
    Returns:
        question_classes: Assignments of labels to questions
            [questions x classes]
    """

    [nQuestions, nParticipants, nClasses] = np.shape(counts)

    question_classes = np.zeros([nQuestions, nClasses])
    final_classes = np.zeros([nQuestions, nClasses])

    for i in range(nQuestions):
        for j in range(nClasses):
            estimate = class_marginals[j]
            estimate *= np.prod(np.power(error_rates[:, j, :], counts[i, :, :]))
            question_classes[i, j] = estimate

        indices = np.argwhere(question_classes[i, :] == np.max(question_classes[i, :])).flatten()
        final_classes[i, np.random.choice(indices)] = 1

    return final_classes

if __name__ == '__main__':
    #datafile = '../data/Generate_data/HDS_generate_date.csv'
    #datafile = r'../data/redundancy_2/train_data_15.csv'
    datafile = r'../data/TDG4Crowd_Text.csv'
    # datafile = '../data/Generate_data/IRT_generate_date.csv'
    truth_value = r'../data/text_truth.csv'

    model = FDS(datafile, truth_value)

    # accs = {}
    # for b in range(450, -1, -10):
    #     datafile = r'../fillData/data/data_fill_' + str(b) + '.csv'
    #     truth_value = r'../data/text_truth.csv'
    #     model, acc = FDS(datafile, truth_value)
    #     data = pd.read_csv(datafile)
    #     label = round(data.shape[0] / 245476, 3)
    #     label = int(label * 1000)
    #
    #     if label not in accs:
    #         accs[str(label)] = acc
    # print(accs)
