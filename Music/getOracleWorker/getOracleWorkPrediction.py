import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from keras import backend as K
from Music.getOracleWorker import accuracy


def multiply(output1, output2):
    output = tf.multiply(output1, output2)
    output = K.sum(output, axis=2)
    return output
def bestWorkerPredict(gate_output):
    '''
        1. 读取文件，筛选出所有的任务[任务编号， 任务特征]
        2. 根据最佳工人的选择权重进行预测
        '''
    # 1. 读取文件，筛选出所有的任务[任务编号， 任务特征]
    datas = pd.read_csv('../data/music_feature.csv', index_col=None)
    datas = datas.drop(columns=['class', 'annotator']).drop_duplicates('id')
    print(datas)
    predict_data = datas.drop(columns=['id'])
    id = datas['id'].reset_index(drop=True)

    # 加载模型
    loaded_model = load_model('../train/musicModel_0.79')
    gate_output = gate_output

    expert_output = loaded_model.layers[2].predict(predict_data)
    out_put = multiply(expert_output, gate_output)
    predict_result = loaded_model.layers[6].predict(out_put)

    predict_result = np.argmax(predict_result, axis=1)
    predict_result = pd.DataFrame({'label': predict_result})
    predict_result = pd.concat([id, predict_result], axis=1)
    pd.DataFrame(predict_result).to_csv('../data/bestWorkerPredict.csv')


    return accuracy.workerAcc()

if __name__ == '__main__':

    '''
    1. 读取文件，筛选出所有的任务[任务编号， 任务特征]
    2. 根据最佳工人的选择权重进行预测
    '''
    # 1. 读取文件，筛选出所有的任务[任务编号， 任务特征]
    datas = pd.read_csv('../data/music_feature.csv', index_col=None)
    datas = datas.drop(columns=['class', 'annotator']).drop_duplicates('id')
    print(datas)
    predict_data = datas.drop(columns=['id'])
    id = datas['id'].reset_index(drop=True)

    # 加载模型
    loaded_model = load_model('../train/musicModel')
    gate_output = [0.5156998865486248, 0.09623719170118047, 0.09537547026873343, 0.10968075721852603,
                   0.009057698687279148, 0.0, 0.01505822583475429, 0.15889076974090197, 0.0, 0.0]

    expert_output = loaded_model.layers[2].predict(predict_data)
    out_put = multiply(expert_output, gate_output)
    predict_result = loaded_model.layers[6].predict(out_put)

    predict_result = np.argmax(predict_result, axis=1)
    predict_result = pd.DataFrame({'label': predict_result})
    predict_result = pd.concat([id, predict_result], axis=1)
    pd.DataFrame(predict_result).to_csv('../data/bestWorkerPredict.csv')

