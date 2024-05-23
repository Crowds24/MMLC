import pandas as pd

def accuracy():
    ground_truth = pd.read_csv('../data/text_truth.csv', index_col=0)

    predict_data = pd.read_csv('../data/bestWorkerPredict.csv', index_col=0)
    print(predict_data)

    truth_dict = {}
    for index, data in ground_truth.iterrows():
        truth_dict[data['task_id']] = data['ground_truth']

    predict_dict = {}
    for index, data in predict_data.iterrows():
        predict_dict[data['task_id']] = data['label']

    count = 0

    for predict in predict_dict:
        if (predict_dict[predict] == truth_dict[predict]):
            count += 1

    print(count / len(predict_dict))
    return count / len(predict_dict)

if __name__ == '__main__':

    ground_truth = pd.read_csv('../data/text_truth.csv', index_col=0)

    predict_data = pd.read_csv('../data/bestWorkerPredict.csv', index_col=0)
    print(predict_data)

    truth_dict = {}
    for index, data in ground_truth.iterrows():

        truth_dict[data['task_id']] = data['ground_truth']

    predict_dict = {}
    for index, data in predict_data.iterrows():
        predict_dict[data['task_id']] = data['label']

    count = 0

    for predict in predict_dict:
        if(predict_dict[predict] == truth_dict[predict]):
            count += 1

    print(count / len(predict_dict))