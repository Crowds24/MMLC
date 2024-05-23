import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

def G_model(b):
    # 一个图片对应399个工人
    # 1. 读取数据
    column_images = ['taskId']
    for i in range(0, 8192):
        column_images.append(str(i))

    column_images_swap = ['taskId', 'worker']
    for i in range(0, 8192):
        column_images_swap.append(str(i))
    column_truth = ['taskId', 'label']
    selected_columns = ['image_1', 'image_2', 'image_3', 'image_4', 'image_5']
    column_names = ['taskId', 'worker', 'label']
    for i in range(0, 8192):
        column_names.append(str(i))
    '''
    准备数据
    读取全部的数据，根据工人标注任务数进行数据的填充。
    '''
    train_data = pd.read_csv('../data/data_train.csv')
    data = train_data

    train_data = train_data[column_names]
    workers = train_data[['worker']].drop_duplicates('worker')
    workers = workers.reset_index(drop=True)

    worker_counts = train_data['worker'].value_counts()

    filter_max_workers = worker_counts[worker_counts > b]
    filter_min_workers = worker_counts[worker_counts <= b]

    workers_1 = pd.get_dummies(workers, columns=['worker'])
    workers_1 = pd.concat([workers, workers_1], axis=1)

    '''工人标注任务数大于n的数据，这些数据进行数据补全'''
    train_min_data = train_data[train_data['worker'].isin(filter_min_workers.index)]
    train_min_data = train_min_data.iloc[:, :3]

    train_min_data.rename(columns={'taskId': 'taskId'}, inplace=True)
    train_data = train_data[train_data['worker'].isin(filter_max_workers.index)]

    truth_data = pd.read_csv(
        '../data/label_train.csv',
        header=None,
        index_col=None,
        names=column_truth
    )
    # 获得真值，并按照task编号排序
    label_truth = truth_data[['taskId', 'label']]
    label_truth = label_truth.sort_values('taskId', ascending=True)


    # 2.1 确定图片数 去掉重复的，并按照task排序
    images = train_data[column_images].drop_duplicates(column_images)

    images = images.sort_values('taskId', ascending=True)

    # 2.2 每张图片对应所有的工人 --找出所有工人
    workers = train_data[['worker']].drop_duplicates('worker')
    workers = workers.reset_index(drop=True)
    workers_one = workers
    workers = pd.merge(workers_1, workers, on='worker', how='inner')
    workers = workers.drop(columns=['worker'])

    # 2.3 预测的数据
    predict_data_train = []
    for i in range(len(images)):
        new_images = pd.DataFrame(images.iloc[[i]].values.repeat(filter_max_workers.shape[0], axis=0),
                                  columns=images.columns)
        new_images = new_images.reset_index(drop=True)
        predict_data_train.append(new_images)

    loaded_model = load_model('../train/modelImage_82')
    # 使用模型进行数据的填充
    results = []

    i = 0
    for result in predict_data_train:
        result = pd.concat([result, workers_one], axis=1)

        result = result[column_images_swap]
        # print(result)

        result['taskId'] = result['taskId'].astype(int)
        # result['worker'] = result['worker'].astype(int)

        result = result.drop(columns=['taskId', 'worker'])
        y = loaded_model.predict([workers, result])

        i = i + 1
        results.append(y)
    # pd.DataFrame(results).to_csv('data/result_predict.csv')
    # 输出填充后的工人回答矩阵
    data_plus = []
    index = 0
    for result in results:

        y_pred = np.argmax(result, axis=1)
        y_pred = pd.DataFrame({'label': y_pred})

        y = pd.concat([workers_one, y_pred], axis=1)
        x = []
        for i in range(filter_max_workers.shape[0]):
            x.append(label_truth[['taskId']].values[index][0])
        x = pd.DataFrame({'taskId': x})
        y = pd.concat([x, y], axis=1)
        index = index + 1
        data_plus.append(y)

    y_plus = pd.concat(data_plus, axis=0, ignore_index=True)
    pd.DataFrame(y_plus).to_csv('../data/data-plus.csv')

    '''
    模型预测的与工人标注的结合在一起
    '''
    column_names = ['id', 'taskId', 'worker', 'label']
    df1 = pd.read_csv('../data/data-plus.csv')
    df1.rename(columns=dict(zip(df1.columns, column_names)), inplace=True)
    df1 = df1.drop(columns=['id'])


    data = data[['taskId', 'worker', 'label']]
    df1 = df1.merge(data[['taskId', 'worker', 'label']], on=['taskId', 'worker'], how='left', suffixes=('_x', ''))
    df1['label'] = df1['label'].fillna(df1['label_x'])
    df1 = df1.drop(['label_x'], axis=1)

    df1 = pd.concat([df1, train_min_data], axis=0)

    df1['taskId'] = df1['taskId'].astype(int)
    df1['label'] = df1['label'].astype(int)

    pd.DataFrame(df1).to_csv('data/data_fill_'+str(b) + '.csv', index=False)

if __name__ == '__main__':

    for b in range(0, 200, 10):
        G_model(b)

    # 一个图片对应399个工人
    # 1. 读取数据
    # column_images = ['taskId']
    # for i in range(0, 8192):
    #     column_images.append(str(i))
    #
    # column_images_swap = ['taskId', 'worker']
    # for i in range(0, 8192):
    #     column_images_swap.append(str(i))
    # column_truth = ['taskId', 'label']
    # selected_columns = ['image_1', 'image_2', 'image_3', 'image_4', 'image_5']
    # column_names = ['taskId', 'worker', 'label']
    # for i in range(0, 8192):
    #     column_names.append(str(i))
    # '''
    # 准备数据
    # 读取全部的数据，根据工人标注任务数进行数据的填充。
    # '''
    # train_data = pd.read_csv('../data/new/train/data_train.csv')
    # data = train_data
    # print(data)
    # train_data = train_data[column_names]
    # workers = train_data[['worker']].drop_duplicates('worker')
    # workers = workers.reset_index(drop=True)
    #
    #
    # worker_counts = train_data['worker'].value_counts()
    # print(worker_counts.values)
    # filter_max_workers = worker_counts[worker_counts > 0]
    # filter_min_workers = worker_counts[worker_counts <= 0]
    # print(filter_min_workers)
    # workers_1 = pd.get_dummies(workers, columns=['worker'])
    # workers_1 = pd.concat([workers, workers_1], axis=1)
    #
    #
    # '''工人标注任务数大于n的数据，这些数据进行数据补全'''
    # train_min_data = train_data[train_data['worker'].isin(filter_min_workers.index)]
    # train_min_data = train_min_data.iloc[:, :3]
    # print(train_min_data)
    # train_min_data.rename(columns={'taskId': 'taskId'}, inplace=True)
    # train_data = train_data[train_data['worker'].isin(filter_max_workers.index)]
    #
    #
    #
    # truth_data = pd.read_csv(
    #     '../data/new/train/label_train.csv',
    #     header=None,
    #     index_col=None,
    #     names=column_truth
    # )
    # # 获得真值，并按照task编号排序
    # label_truth = truth_data[['taskId', 'label']]
    # label_truth = label_truth.sort_values('taskId', ascending=True)
    # print(label_truth)
    #
    #
    #
    # # 2.1 确定图片数 去掉重复的，并按照task排序
    # images = train_data[column_images].drop_duplicates(column_images)
    # print(images)
    # images = images.sort_values('taskId', ascending=True)
    # print(images)
    # # 2.2 每张图片对应所有的工人 --找出所有工人
    # workers = train_data[['worker']].drop_duplicates('worker')
    # workers = workers.reset_index(drop=True)
    # workers_one = workers
    # workers = pd.merge(workers_1, workers, on='worker', how='inner')
    # workers = workers.drop(columns=['worker'])
    #
    # # 2.3 预测的数据
    # predict_data_train = []
    # for i in range(len(images)):
    #     new_images = pd.DataFrame(images.iloc[[i]].values.repeat(filter_max_workers.shape[0], axis=0), columns=images.columns)
    #     new_images = new_images.reset_index(drop=True)
    #     predict_data_train.append(new_images)
    #
    #
    # loaded_model = load_model('../train/modelImage_origin')
    # # 使用模型进行数据的填充
    # results = []
    # print(len(predict_data_train))
    # i = 0
    # for result in predict_data_train:
    #     result = pd.concat([result, workers_one], axis=1)
    #
    #     result = result[column_images_swap]
    #     #print(result)
    #
    #     result['taskId'] = result['taskId'].astype(int)
    #     # result['worker'] = result['worker'].astype(int)
    #
    #     result = result.drop(columns=['taskId', 'worker'])
    #     y = loaded_model.predict([workers, result])
    #     print(i)
    #     i = i+1
    #     results.append(y)
    # # pd.DataFrame(results).to_csv('data/result_predict.csv')
    # # 输出填充后的工人回答矩阵
    # data_plus = []
    # index = 0
    # for result in results:
    #
    #     y_pred = np.argmax(result, axis=1)
    #     y_pred = pd.DataFrame({'label': y_pred})
    #
    #     y = pd.concat([workers_one, y_pred], axis=1)
    #     x = []
    #     for i in range(filter_max_workers.shape[0]):
    #         x.append(label_truth[['taskId']].values[index][0])
    #     x = pd.DataFrame({'taskId': x})
    #     y = pd.concat([x, y], axis=1)
    #     index = index + 1
    #     data_plus.append(y)
    #
    #
    # y_plus = pd.concat(data_plus, axis=0, ignore_index=True)
    # pd.DataFrame(y_plus).to_csv('../data/data-plus.csv')
    #
    # '''
    # 模型预测的与工人标注的结合在一起
    # '''
    # column_names = ['id', 'taskId', 'worker', 'label']
    # df1 = pd.read_csv('../data/data-plus.csv')
    # df1.rename(columns=dict(zip(df1.columns, column_names)), inplace=True)
    # df1 = df1.drop(columns=['id'])
    # print(df1)
    #
    # data = data[['taskId', 'worker', 'label']]
    # df1 = df1.merge(data[['taskId', 'worker', 'label']], on=['taskId', 'worker'], how='left', suffixes=('_x', ''))
    # df1['label'] = df1['label'].fillna(df1['label_x'])
    # df1 = df1.drop(['label_x'], axis=1)
    # print(df1)
    # print(filter_min_workers)
    # df1 = pd.concat([df1, train_min_data], axis=0)
    # print(df1)
    # df1['taskId'] = df1['taskId'].astype(int)
    # df1['label'] = df1['label'].astype(int)
    # print(df1)
    # pd.DataFrame(df1).to_csv('data/data-plus_1.csv', index=False)

