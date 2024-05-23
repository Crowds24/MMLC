import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from keras import backend as K

from Text.getOracleWorker import accuracy


def multiply(output1, output2):
    output = tf.multiply(output1, output2)
    output = K.sum(output, axis=2)
    return output

def bestWorkerPredict(gate_output):
    '''
        1. 读取文件，筛选出所有的任务[任务编号， 任务特征]
        2. 根据最佳工人的选择权重进行预测
        '''
    column_images_swap = ['task_id', 'annotator_id']

    column_names = ['task_id', 'annotator_id', 'answer']
    for i in range(1, 769):
        column_names.append("feature_" + str(i))
    '''
    准备数据
    读取全部的数据，根据工人标注任务数进行数据的填充。
    '''
    dates = pd.read_csv('../data/data_BERT.csv', index_col=None)
    print(dates)
    datas = dates.drop(columns=['annotator_id', 'answer']).drop_duplicates('task_id')
    predict_data = datas.drop(columns=['task_id'])
    task_id = datas['task_id'].reset_index(drop=True)

    # 加载模
    loaded_model = load_model('../train/textModel_origin')
    gate_output = gate_output

    expert_output = loaded_model.layers[2].predict(predict_data)
    out_put = multiply(expert_output, gate_output)
    predict_result = loaded_model.layers[6].predict(out_put)

    predict_result = np.argmax(predict_result, axis=1)
    predict_result = pd.DataFrame({'label': predict_result})
    predict_result = pd.concat([task_id, predict_result], axis=1)
    pd.DataFrame(predict_result).to_csv('../data/bestWorkerPredict.csv')

    return accuracy.accuracy()

if __name__ == '__main__':

    '''
    1. 读取文件，筛选出所有的任务[任务编号， 任务特征]
    2. 根据最佳工人的选择权重进行预测
    '''
    column_images_swap = ['task_id', 'annotator_id']

    column_names = ['task_id', 'annotator_id', 'answer']
    for i in range(1, 769):
        column_names.append("feature_" + str(i))
    '''
    准备数据
    读取全部的数据，根据工人标注任务数进行数据的填充。
    '''
    dates = pd.read_csv('../data/data_BERT.csv', index_col=None)
    print(dates)
    datas = dates.drop(columns=['annotator_id', 'answer']).drop_duplicates('task_id')
    predict_data = datas.drop(columns=['task_id'])
    task_id = datas['task_id'].reset_index(drop=True)

    # 加载模
    loaded_model = load_model('../train/textModel')
    gate_output = [0.1630767496738678, 0.07045636047954941, 0.02553008954989861, 0.164726254270243, 0.13454994123110456, 0.06295374268764005, 0.12880569763266214,
                   0.053657029130898544, 0.10766339677934782, 0.08858073856478811]

    expert_output = loaded_model.layers[2].predict(predict_data)
    out_put = multiply(expert_output, gate_output)
    predict_result = loaded_model.layers[6].predict(out_put)

    predict_result = np.argmax(predict_result, axis=1)
    predict_result = pd.DataFrame({'label': predict_result})
    predict_result = pd.concat([task_id, predict_result], axis=1)
    pd.DataFrame(predict_result).to_csv('../data/bestWorkerPredict.csv')

