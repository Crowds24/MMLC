import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.neighbors import KernelDensity

from Text.getOracleWorker import getOracleWorkPrediction
from Text.getOracleWorker.getOracleWorkPrediction import bestWorkerPredict
from Text.until import plotWorkerScatter


# 根据工人标注的数量技术权重

def annotator_id_weight(datas):
    annotator_id_counts = {}

    for index, data in datas.iterrows():
        annotator_id = data[1]
        if annotator_id in annotator_id_counts:
            annotator_id_counts[int(annotator_id)] += 1
        else:
            annotator_id_counts[int(annotator_id)] = 1

    print(annotator_id_counts)

    weights = {}

    total_counts = sum(annotator_id_counts.values())
    print("total_counts", total_counts)

    for annotator_id, count in annotator_id_counts.items():
        weight = count / total_counts
        weights[annotator_id] = weight
    print(weights)

    return weights

# 计算权重b
def claculate_b(data_values, mean_data):

    matrix_annotator_id = np.array(data_values)
    matrix_oralWorker = np.tile(mean_data, (70, 1))

    print(matrix_annotator_id.shape)
    print(matrix_oralWorker.shape)

    matrix_annotator_id_1 = np.linalg.pinv(matrix_annotator_id)
    print(matrix_annotator_id_1.shape)

    b = np.matmul(np.linalg.pinv(matrix_annotator_id), matrix_oralWorker)
    #b = np.matmul(matrix_oralWorker, np.linalg.pinv(matrix_annotator_id))

    print(b)

def density_media(data_values):
    density_medias = []
    transposed_data = np.transpose(data_values)
    for data in transposed_data:
        data = np.array(data).reshape(-1, 1)
        # bandwidth=0.001 ACC=0.748  0.0005 75 1 0.73
        kde = KernelDensity(bandwidth=0.001, kernel='gaussian')
        kde.fit(data)
        x_values = np.linspace(np.min(data), np.max(data), num=100).reshape(-1, 1)
        demsities = np.exp(kde.score_samples(x_values))
        max_density_index = np.argmax(demsities)
        media_estimate = x_values[max_density_index][0]
        density_medias.append(media_estimate)
        print(media_estimate)
    return density_medias

if __name__ == '__main__':
    # 一个图片对应399个工人
    # 1. 读取数据

    column_images_swap = ['task_id', 'annotator_id']

    column_names = ['task_id', 'annotator_id', 'answer']
    for i in range(1, 769):
        column_names.append("feature_" + str(i))
    '''
    准备数据
    读取全部的数据，根据工人标注任务数进行数据的填充。
    '''
    train_data = pd.read_csv('../data/data_BERT.csv', index_col=0)
    print(train_data)
    annotator_ids = train_data[['annotator_id']].drop_duplicates('annotator_id')
    annotator_ids = annotator_ids.reset_index(drop=True)
    train_data_gate = pd.get_dummies(annotator_ids, columns=['annotator_id'])
    loaded_model = load_model('../train/textModel_origin')
    data = loaded_model.layers[3].predict(train_data_gate)
    print(data[0].shape)
    data_values = []
    for i in range(data.shape[0]):
        sub_data = data[i]
        data_values.append(sub_data[0])
    print(data_values)

    # workers_acc = []
    #
    # for worker_select in data_values:
    #     acc = getBestWorkPrediction.bestWorkerPredict(worker_select)
    #     workers_acc.append(acc)
    # print(workers_acc)




    # weights = annotator_id_weight(train_data)
    # annotator_ids = annotator_ids['annotator_id']
    # print(annotator_ids)
    # data_weighted = []
    #
    # for annotator_id, data_value in zip(annotator_ids, data_values):
    #     data = weights[annotator_id] * data_value
    #     data_weighted.append(data)

    mean_data = density_media(data_values)

    # 计算均值
    # mean_data = np.mean(data_values, axis=0)
    sum_data = sum(mean_data)
    data = []
    for i in mean_data:
        data.append(i / sum_data)
    #
    print(mean_data)
    print(data)
    print(bestWorkerPredict(data))


    # 散点图
    # oracle_worker = [0.053305521494511406, 0.07019503908514073, 0.031879648603875224, 0.07694784924387932,
    #                  0.04160289917461047, 0.04587434254812472,
    #                  0.09807082265615463, 0.049070558915234576, 0.08418243420733647, 0.11319732726222337]
    # # numpy_array = np.array(data_values)
    # numpy_array = np.load('numpy_array.npy')
    # # numpy_array = np.vstack((numpy_array, oracle_worker))
    # print(numpy_array)
    # # plotWorkerScatter.workerScatter_t_SNE(numpy_array)
    # # plotWorkerScatter.workerScatter(numpy_array)
    # # plotWorkerScatter.workerScatter_ICA(numpy_array)
    # # plotWorkerScatter.workerScatter_KPCA(numpy_array)
    # # plotWorkerScatter.workerScatterFactor(numpy_array)
    # plotWorkerScatter.workerScatter_IOSMAP_3D(numpy_array)

