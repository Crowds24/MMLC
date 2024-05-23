import numpy as np
import pandas as pd

def accuracy():
    ground_truth = np.load('../data/labels_train.npy')
    ground_truth = pd.DataFrame(ground_truth)
    ground_truth.reset_index(level=0, inplace=True)
    ground_truth.rename(columns={'index': 'taskId'}, inplace=True)
    predict_data = pd.read_csv('../data/bestWorkerPredict.csv', index_col=0)

    truth_dict = {}
    for index, data in ground_truth.iterrows():
        truth_dict[data['taskId']] = data[0]

    predict_dict = {}
    for index, data in predict_data.iterrows():
        predict_dict[data['taskId']] = data['label']

    count = 0

    for predict in predict_dict:
        if (predict_dict[predict] == truth_dict[predict]):
            count += 1
    print(count / len(predict_dict))
    return count / len(predict_dict)
if __name__ == '__main__':

    ground_truth = np.load('../data/labels_train.npy')
    ground_truth = pd.DataFrame(ground_truth)
    ground_truth.reset_index(level=0, inplace=True)
    ground_truth.rename(columns={'index': 'taskId'}, inplace=True)
    predict_data = pd.read_csv('../data/bestWorkerPredict.csv', index_col=0)

    truth_dict = {}
    for index, data in ground_truth.iterrows():
        truth_dict[data['taskId']] = data[0]

    predict_dict = {}
    for index, data in predict_data.iterrows():
        predict_dict[data['taskId']] = data['label']

    count = 0

    for predict in predict_dict:
        if(predict_dict[predict] == truth_dict[predict]):
            count += 1
    print(count / len(predict_dict))