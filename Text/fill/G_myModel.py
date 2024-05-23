import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

def G_model(b):
    #  # 一个图片对应399个工人
    # 1. 读取数据
    column_images = ['task_id']
    for i in range(1, 769):
        column_images.append("feature_" + str(i))

    column_images_swap = ['task_id', 'annotator_id']
    for i in range(1, 769):
        column_images_swap.append("feature_" + str(i))

    column_truth = ['task_id', 'ground_truth']

    column_names = ['task_id', 'annotator_id', 'answer']
    for i in range(1, 769):
        column_names.append("feature_" + str(i))
    '''
    准备数据
    读取全部的数据，根据工人标注任务数进行数据的填充。
    '''
    train_data = pd.read_csv('../data/data_BERT.csv', index_col=None, header=0, names=column_names)

    workers = train_data[['annotator_id']].drop_duplicates('annotator_id')
    workers = workers.reset_index(drop=True)

    m = b

    train_data_1 = train_data
    worker_counts = train_data['annotator_id'].value_counts()
    print(worker_counts.values)
    filter_max_workers = worker_counts[worker_counts > m]
    filter_min_workers = worker_counts[worker_counts <= m]
    workers_1 = pd.get_dummies(workers, columns=['annotator_id'])
    workers_1 = pd.concat([workers, workers_1], axis=1)

    '''工人标注任务数大于n的数据，这些数据进行数据补全'''
    train_min_data = train_data[train_data['annotator_id'].isin(filter_min_workers.index)]
    train_min_data = train_min_data.iloc[:, :3]
    train_data = train_data[train_data['annotator_id'].isin(filter_max_workers.index)]

    truth_data = pd.read_csv(
        '../data/text_truth.csv',
        header=0,
        index_col=0,
    )

    # 获得真值，并按照task编号排序
    truth_data = truth_data.drop_duplicates('task_id')

    label_truth = truth_data.sort_values('task_id', ascending=True)


    # 2.使用一个列表存放数据：[图片：399个工人]
    # 2.1 确定图片数 去掉重复的，并按照task排序
    images = train_data_1[column_images].drop_duplicates(column_images)
    images = images.sort_values('task_id', ascending=True)

    # 2.2 每张图片对应所有的工人 --找出所有工人
    workers = train_data[['annotator_id']].drop_duplicates('annotator_id')
    workers = workers.reset_index(drop=True)

    workers_one = workers
    workers = pd.merge(workers_1, workers, on='annotator_id', how='inner')
    workers = workers.drop(columns=['annotator_id'])

    # workers = pd.get_dummies(workers, columns=['worker'])
    # 2.3 预测的数据
    predict_data_train = []
    for i in range(len(images)):
        new_images = pd.DataFrame(images.iloc[[i]].values.repeat(filter_max_workers.shape[0], axis=0),
                                  columns=images.columns)
        new_images = new_images.reset_index(drop=True)
        predict_data_train.append(new_images)

    loaded_model = load_model('../train/textModel_origin')
    # 使用模型进行数据的填充
    results = []

    i = 0
    for result in predict_data_train:
        result = pd.concat([result, workers_one], axis=1)

        result = result[column_images_swap]
        # print(result)

        result['task_id'] = result['task_id'].astype(int)
        result = result.drop(columns=['task_id', 'annotator_id'])
        print(result.shape)
        y = loaded_model.predict([workers, result])

        i = i + 1
        results.append(y)

    # 输出填充后的工人回答矩阵
    data_plus = []
    index = 0
    for result in results:

        y_pred = np.argmax(result, axis=1)
        y_pred = pd.DataFrame({'label': y_pred})

        y = pd.concat([workers_one, y_pred], axis=1)
        # print(y)
        x = []
        for i in range(filter_max_workers.shape[0]):
            x.append(label_truth[['task_id']].values[index][0])
        x = pd.DataFrame({'task_id': x})
        y = pd.concat([x, y], axis=1)
        index = index + 1
        data_plus.append(y)

    y_plus = pd.concat(data_plus, axis=0, ignore_index=True)
    pd.DataFrame(y_plus).to_csv('data/data-plus.csv')

    '''
    模型预测的与工人标注的结合在一起
    '''
    column_names = ['id', 'task_id', 'annotator_id', 'answer']
    df1 = pd.read_csv('data/data-plus.csv')
    df1.rename(columns=dict(zip(df1.columns, column_names)), inplace=True)
    df1 = df1.drop(columns=['id'])

    column_names = ['task_id', 'annotator_id', 'answer']
    for i in range(1, 769):
        column_names.append("feature_" + str(i))
    df2 = pd.read_csv(
        '../data/data_BERT.csv', index_col=None, header=0, names=column_names
    )
    df2 = df2[['task_id', 'annotator_id', 'answer']]

    df1 = df1.merge(df2[['task_id', 'annotator_id', 'answer']], on=['task_id', 'annotator_id'], how='left',
                    suffixes=('_x', ''))
    df1['answer'] = df1['answer'].fillna(df1['answer_x'])
    df1 = df1.drop(['answer_x'], axis=1)
    df1 = pd.concat([df1, train_min_data], axis=0)

    df1['answer'] = df1['answer'].astype(int)
    pd.DataFrame(df1).to_csv('data/data_fill_' + str(m) + '.csv', index=None)



if __name__ == '__main__':
    for b in range(0, 1150, 10):
        G_model(b)
   # #  # 一个图片对应399个工人
   #  # 1. 读取数据
   #  column_images = ['task_id']
   #  for i in range(1, 769):
   #      column_images.append("feature_" + str(i))
   #
   #  column_images_swap = ['task_id', 'annotator_id']
   #  for i in range(1, 769):
   #      column_images_swap.append("feature_" + str(i))
   #
   #  column_truth = ['task_id', 'ground_truth']
   #
   #  column_names = ['task_id', 'annotator_id', 'answer']
   #  for i in range(1, 769):
   #      column_names.append("feature_" + str(i))
   #  '''
   #  准备数据
   #  读取全部的数据，根据工人标注任务数进行数据的填充。
   #  '''
   #  train_data = pd.read_csv('../data/data_BERT.csv', index_col=None, header=0, names=column_names)
   #  print(train_data)
   #  workers = train_data[['annotator_id']].drop_duplicates('annotator_id')
   #  workers = workers.reset_index(drop=True)
   #
   #  m = 0
   #  task = train_data[['task_id']].drop_duplicates(subset=['task_id'])
   #  train_data_1 = train_data
   #  worker_counts = train_data['annotator_id'].value_counts()
   #  print('worker_counts', worker_counts.values)
   #  filter_max_workers = worker_counts[worker_counts > m]
   #  print('filter_max_workers', filter_max_workers)
   #  filter_min_workers = worker_counts[worker_counts <= m]
   #  workers_1 = pd.get_dummies(workers, columns=['annotator_id'])
   #  workers_1 = pd.concat([workers, workers_1], axis=1)
   #
   #
   #  '''工人标注任务数大于n的数据，这些数据进行数据补全'''
   #  train_min_data = train_data[train_data['annotator_id'].isin(filter_min_workers.index)]
   #  train_min_data = train_min_data.iloc[:, :3]
   #  train_data = train_data[train_data['annotator_id'].isin(filter_max_workers.index)]
   #
   #
   #
   #  truth_data = pd.read_csv(
   #      '../data/text_truth.csv',
   #      header=0,
   #      index_col=0,
   #  )
   #  print(truth_data)
   #  # 获得真值，并按照task编号排序
   #  truth_data = truth_data.drop_duplicates('task_id')
   #  print(truth_data)
   #  label_truth = truth_data.sort_values('task_id', ascending=True)
   #  print("label:", label_truth)
   #
   #  # 2.使用一个列表存放数据：[图片：399个工人]
   #  # 2.1 确定图片数 去掉重复的，并按照task排序
   #  images = train_data_1[column_images].drop_duplicates(column_images)
   #  images = images.sort_values('task_id', ascending=True)
   #  print(images)
   #  # 2.2 每张图片对应所有的工人 --找出所有工人
   #  workers = train_data[['annotator_id']].drop_duplicates('annotator_id')
   #  workers = workers.reset_index(drop=True)
   #  print("worker:", workers)
   #  workers_one = workers
   #  workers = pd.merge(workers_1, workers, on='annotator_id', how='inner')
   #  workers = workers.drop(columns=['annotator_id'])
   #  print(workers)
   # # workers = pd.get_dummies(workers, columns=['worker'])
   #  # 2.3 预测的数据
   #  predict_data_train = []
   #  for i in range(len(images)):
   #      new_images = pd.DataFrame(images.iloc[[i]].values.repeat(filter_max_workers.shape[0], axis=0), columns=images.columns)
   #      new_images = new_images.reset_index(drop=True)
   #      predict_data_train.append(new_images)
   #
   #
   #  loaded_model = load_model('model_BERT(1)')
   #  # 使用模型进行数据的填充
   #  results = []
   #  print(len(predict_data_train))
   #  i = 0
   #  for result in predict_data_train:
   #      result = pd.concat([result, workers_one], axis=1)
   #
   #      result = result[column_images_swap]
   #      #print(result)
   #
   #      result['task_id'] = result['task_id'].astype(int)
   #      result = result.drop(columns=['task_id', 'annotator_id'])
   #      y = loaded_model.predict([workers, result])
   #      print(i)
   #      i = i+1
   #      results.append(y)
   #
   #  # 输出填充后的工人回答矩阵
   #  data_plus = []
   #  index = 0
   #  for result in results:
   #
   #      y_pred = np.argmax(result, axis=1)
   #      y_pred = pd.DataFrame({'label': y_pred})
   #
   #      y = pd.concat([workers_one, y_pred], axis=1)
   #      #print(y)
   #      x = []
   #      for i in range(filter_max_workers.shape[0]):
   #          x.append(label_truth[['task_id']].values[index][0])
   #      x = pd.DataFrame({'task_id': x})
   #      y = pd.concat([x, y], axis=1)
   #      index = index + 1
   #      data_plus.append(y)
   #
   #
   #  y_plus = pd.concat(data_plus, axis=0, ignore_index=True)
   #  pd.DataFrame(y_plus).to_csv('data/data-plus.csv')
   #
   #  '''
   #  模型预测的与工人标注的结合在一起
   #  '''
   #  column_names = ['id', 'task_id', 'annotator_id', 'answer']
   #  df1 = pd.read_csv('data/data-plus.csv')
   #  df1.rename(columns=dict(zip(df1.columns, column_names)), inplace=True)
   #  df1 = df1.drop(columns=['id'])
   #  print(df1)
   #  column_names = ['task_id', 'annotator_id', 'answer']
   #  for i in range(1, 769):
   #      column_names.append("feature_" + str(i))
   #  df2 = pd.read_csv(
   #      '../data/data_BERT.csv', index_col=None, header=0, names=column_names
   #  )
   #  df2 = df2[['task_id', 'annotator_id', 'answer']]
   #  print(df2)
   #  df1 = df1.merge(df2[['task_id', 'annotator_id', 'answer']], on=['task_id', 'annotator_id'], how='left', suffixes=('_x', ''))
   #  df1['answer'] = df1['answer'].fillna(df1['answer_x'])
   #  df1 = df1.drop(['answer_x'], axis=1)
   #  df1 = pd.concat([df1, train_min_data], axis=0)
   #
   #  df1['answer'] = df1['answer'].astype(int)
   #  pd.DataFrame(df1).to_csv('data/data-plus_BTER_1'+str(m)+'.csv', index=None)

