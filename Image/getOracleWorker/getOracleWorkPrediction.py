import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from keras import backend as K

from Image.getOracleWorker import accuracy


def multiply(output1, output2):
    output = tf.multiply(output1, output2)
    output = K.sum(output, axis=2)
    return output

def bestWorkerPredict(data):
    '''
        1. 读取文件，筛选出所有的任务[任务编号， 任务特征]
        2. 根据最佳工人的选择权重进行预测
        '''

    # 1. 读取文件，筛选出所有的任务[任务编号， 任务特征]
    datas = pd.read_csv('../data/data_train.csv', index_col=None)

    datas = datas.drop_duplicates('taskId')
    predict_data = datas.drop(columns=['taskId', 'worker', 'label'])
    taskId = datas['taskId'].reset_index(drop=True)

    # 加载模型
    loaded_model = load_model('../train/modelImage_origin')
    gate_output = data

    expert_output = loaded_model.layers[2].predict(predict_data)
    out_put = multiply(expert_output, gate_output)
    predict_result = loaded_model.layers[6].predict(out_put)

    predict_result = np.argmax(predict_result, axis=1)
    predict_result = pd.DataFrame({'label': predict_result})
    predict_result = pd.concat([taskId, predict_result], axis=1)

    pd.DataFrame(predict_result).to_csv('../data/bestWorkerPredict.csv')

    return accuracy.accuracy()

if __name__ == '__main__':

    '''
    1. 读取文件，筛选出所有的任务[任务编号， 任务特征]
    2. 根据最佳工人的选择权重进行预测
    '''

    # 1. 读取文件，筛选出所有的任务[任务编号， 任务特征]
    datas = pd.read_csv('../data/data_train.csv', index_col=None)
    print(datas.shape)
    datas = datas.drop_duplicates('taskId')
    predict_data = datas.drop(columns=['taskId', 'worker', 'label'])
    print(predict_data)
    taskId = datas['taskId'].reset_index(drop=True)

    # 加载模型
    loaded_model = load_model('../train/modelImage')
    gate_output = [0.07124265867938012, 0.044333722173375736, 0.04408480962106964, 0.08786190566702846, 0.04642905555366911, 0.08254022406283161, 0.06239921964599325, 0.06470486201036646, 0.05587863902087971, 0.07447023733547808, 0.0675003711826729, 0.06538015636484727,
                   0.06581017895069745, 0.05599092624674033, 0.06521697099230073, 0.046156062492669]

    expert_output = loaded_model.layers[2].predict(predict_data)
    out_put = multiply(expert_output, gate_output)
    predict_result = loaded_model.layers[6].predict(out_put)

    predict_result = np.argmax(predict_result, axis=1)
    predict_result = pd.DataFrame({'label': predict_result})
    predict_result = pd.concat([taskId, predict_result], axis=1)
    print(predict_result.shape)
    pd.DataFrame(predict_result).to_csv('../data/bestWorkerPredict.csv')

