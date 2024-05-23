import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.neighbors import KernelDensity

from Image.getOracleWorker import getOracleWorkPrediction
from Image.getOracleWorker.getOracleWorkPrediction import bestWorkerPredict
from Image.until import plotWorkerScatter


# 根据工人标注的数量技术权重

def worker_weight(datas):
    worker_counts = {}

    for index, data in datas.iterrows():
        worker = data[1]
        if worker in worker_counts:
            worker_counts[int(worker)] += 1
        else:
            worker_counts[int(worker)] = 1

    print(worker_counts)

    weights = {}

    total_counts = sum(worker_counts.values())
    print("total_counts", total_counts)

    for worker, count in worker_counts.items():
        weight = count / total_counts
        weights[worker] = weight
    print(weights)

    return weights

# 计算权重b
def claculate_b(data_values, mean_data):

    matrix_worker = np.array(data_values)
    matrix_oralWorker = np.tile(mean_data, (70, 1))

    print(matrix_worker.shape)
    print(matrix_oralWorker.shape)

    matrix_worker_1 = np.linalg.pinv(matrix_worker)
    print(matrix_worker_1.shape)

    b = np.matmul(np.linalg.pinv(matrix_worker), matrix_oralWorker)
    #b = np.matmul(matrix_oralWorker, np.linalg.pinv(matrix_worker))

    print(b)

def density_media(data_values):
    density_medias = []
    transposed_data = np.transpose(data_values)
    for data in transposed_data:
        data = np.array(data).reshape(-1, 1)
        # bandwidth=0.05 ACC=0.81
        kde = KernelDensity(bandwidth=0.05, kernel='gaussian')
        kde.fit(data)
        x_values = np.linspace(np.min(data), np.max(data), num=len(data)).reshape(-1, 1)
        demsities = np.exp(kde.score_samples(x_values))
        max_density_index = np.argmax(demsities)
        media_estimate = x_values[max_density_index][0]
        density_medias.append(media_estimate)
        print(media_estimate)
    return density_medias

if __name__ == '__main__':
    '''
    准备数据
    读取全部的数据，根据工人标注任务数进行数据的填充。
    '''
    train_data = pd.read_csv('../data/data_train.csv', index_col=0)
    workers = train_data[['worker']].drop_duplicates('worker')
    workers = workers.reset_index(drop=True)
    train_data_gate = pd.get_dummies(workers, columns=['worker'])
    loaded_model = load_model('../train/modelImage_origin')
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
    #
    # # 散点图
    # numpy_array = np.array(data_values)

    # weights = worker_weight(train_data)
    # workers = workers['worker']
    # print(workers)
    # data_weighted = []
    # for worker, data_value in zip(workers, data_values):
    #     data = weights[worker] * data_value
    #     data_weighted.append(data)
    # 核密度
    mean_data = density_media(data_values)
    # 计算均值
    # mean_data = np.mean(data_values, axis=0)
    # 计算中位数
    # mean_data = np.median(data_values, axis=0)
    sum_data = sum(mean_data)
    data = []
    for i in mean_data:
        data.append(i / sum_data)

    print(mean_data)
    print(data)
    acc = bestWorkerPredict(data)
    print(acc)
    # # 加载保存的NumPy数组
    # oracle_worker = [0.1274182411880429, 0.10120543142093887, 0.10566259603864457, 0.11069937425046995,
    #                  0.12472410703131681, 0.04037496737383016,
    #                  0.07515931281861461, 0.12534738671061416, 0.11218173187960483, 0.07722685128792318]
    # numpy_array = np.load('numpy_array.npy')
    # print(numpy_array.shape)
    # numpy_array = np.vstack((numpy_array, oracle_worker))
    # # plotWorkerScatter.workerScatter(numpy_array)
    # plotWorkerScatter.workerScatter_t_SNE(numpy_array)
    # plotWorkerScatter.workerScatter_ICA(numpy_array)
    # plotWorkerScatter.workerScatter_KPCA(numpy_array)
    # plotWorkerScatter.workerScatter_IOSMAP(numpy_array)
    # plotWorkerScatter.workerScatterFactor(numpy_array)
    # plotWorkerScatter.workerScatterLLE(numpy_array)


