import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.neighbors import KernelDensity

from Music.getOracleWorker import getOracleWorkPrediction
from Music.getOracleWorker.getOracleWorkPrediction import bestWorkerPredict
from Music.until import plotWorkerScatter


# 根据工人标注的数量技术权重

def annotator_weight(datas):
    annotator_counts = {}

    for index, data in datas.iterrows():
        annotator = data[1]
        if annotator in annotator_counts:
            annotator_counts[int(annotator)] += 1
        else:
            annotator_counts[int(annotator)] = 1

    print(annotator_counts)

    weights = {}

    total_counts = sum(annotator_counts.values())
    print("total_counts", total_counts)

    for annotator, count in annotator_counts.items():
        weight = count / total_counts
        weights[annotator] = weight
    print(weights)

    return weights

# 计算权重b
def claculate_b(data_values, mean_data):

    matrix_annotator = np.array(data_values)
    matrix_oralWorker = np.tile(mean_data, (70, 1))

    print(matrix_annotator.shape)
    print(matrix_oralWorker.shape)

    matrix_annotator_1 = np.linalg.pinv(matrix_annotator)
    print(matrix_annotator_1.shape)

    b = np.matmul(np.linalg.pinv(matrix_annotator), matrix_oralWorker)
    #b = np.matmul(matrix_oralWorker, np.linalg.pinv(matrix_annotator))

    print(b)

def density_media(data_values):
    density_medias = []
    transposed_data = np.transpose(data_values)
    for data in transposed_data:
        data = np.array(data).reshape(-1, 1)
        # bandwidth=1 ACC=0.79
        kde = KernelDensity(bandwidth=1, kernel='gaussian')
        kde.fit(data)
        x_values = np.linspace(np.min(data), np.max(data), num=len(data)).reshape(-1, 1)
        demsities = np.exp(kde.score_samples(x_values))
        max_density_index = np.argmax(demsities)
        media_estimate = x_values[max_density_index][0]
        density_medias.append(media_estimate)
        print(media_estimate)
    return density_medias

def normalization(data):
    sum_data = sum(data)
    data = []
    for i in data:
        data.append(i / sum_data)
    print(data)
    return data

if __name__ == '__main__':

    '''
    准备数据
    读取全部的数据，根据工人标注任务数进行数据的填充。
    '''
    train_data = pd.read_csv('../data/music_feature.csv', index_col=None)
    annotators = train_data[['annotator']].drop_duplicates('annotator')
    annotators = annotators.reset_index(drop=True)
    train_data_gate = pd.get_dummies(annotators, columns=['annotator'])
    loaded_model = load_model('../train/musicModel_0.79')
    data = loaded_model.layers[3].predict(train_data_gate)
    data_values = []
    for i in range(data.shape[0]):
        sub_data = data[i]
        data_values.append(sub_data[0])
    print(data_values)

    # workers_acc = []
    #
    # for worker_select in data_values:
    #     acc = getBestWorkPrediction.bestWorkerPredict_C(worker_select)
    #     workers_acc.append(acc)
    # print(workers_acc)
    # numpy_array = np.array(data_values)
    # # 保存NumPy数组
    # np.save('numpy_array.npy', numpy_array)

    # weights = annotator_weight(train_data)
    # annotators = annotators['annotator']
    # print(annotators)
    # data_weighted = []
    # for annotator, data_value in zip(annotators, data_values):
    #     data = weights[annotator] * data_value
    #     data_weighted.append(data)
    #
    mean_data = density_media(data_values)

    # 计算均值
    #mean_data = np.mean(data_values, axis=0)
    #mean_data = np.median(data_values, axis=0)
    sum_data = sum(mean_data)
    data = []
    for i in mean_data:
        data.append(i / sum_data)
    #
    print(mean_data)
    print(data)

    print(bestWorkerPredict(data))


    # 加载保存的NumPy数组
    # oracle_worker = [0.13820427203489055, 0.17980818216228567, 0.10761719211584557, 0.038788771730331156, 0.029946872563708732,
    #                0.2463344734515049, 0.08802976167845389, 0.06770221475833456, 0.09193783730082672, 0.01163042220381843]
    # numpy_array = np.load('numpy_array.npy')
    # print(numpy_array.shape)
    # numpy_array = np.vstack((numpy_array, oracle_worker))
    # # plotWorkerScatter.workerScatter(numpy_array)
    # # plotWorkerScatter.workerScatter_KPCA(numpy_array)
    # # plotWorkerScatter.workerScatter_ICA(numpy_array)
    # # plotWorkerScatter.workerScatter_t_SNE(numpy_array)
    # plotWorkerScatter.workerScatter_IOSMAP(numpy_array)
    # # plotWorkerScatter.workerScatterFactor(numpy_array)
    # # plotWorkerScatter.workerScatterLLE(numpy_array)
    # # plotWorkerScatter.workerScatter_umap(numpy_array)
    # # plotWorkerScatter.workerScatter_mds(numpy_array)


