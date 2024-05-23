import numpy as np
import pandas as pd
from keras.models import load_model

def G_model(b):
    # 一个图片对应399个工人
    # 1. 读取数据
    column_images = ['id']
    for i in range(0, 124):
        column_images.append("feature" + str(i))

    column_images_swap = ['id', 'annotator']
    for i in range(0, 124):
        column_images_swap.append("feature" + str(i))
    column_truth = ['Input.song', 'Input.true_label']
    selected_columns = ['image_1', 'image_2', 'image_3', 'image_4', 'image_5']
    # train_data = pd.read_csv('data/filtered_df_20.csv', index_col=None)
    column_names = ['id', 'annotator', 'class']
    for i in range(0, 124):
        column_names.append("feature" + str(i))
    '''
    准备数据
    读取全部的数据，根据工人标注任务数进行数据的填充。
    '''
    train_data = pd.read_csv('../data/music_feature.csv', index_col=None)
    train_data = train_data[column_names]
    annotators = train_data[['annotator']].drop_duplicates('annotator')
    annotators = annotators.reset_index(drop=True)

    b = b

    annotator_counts = train_data['annotator'].value_counts()
    print(annotator_counts.values)
    filter_max_annotators = annotator_counts[annotator_counts > b]
    filter_min_annotators = annotator_counts[annotator_counts <= b]

    annotators_1 = pd.get_dummies(annotators, columns=['annotator'])
    annotators_1 = pd.concat([annotators, annotators_1], axis=1)

    '''工人标注任务数大于n的数据，这些数据进行数据补全'''
    train_min_data = train_data[train_data['annotator'].isin(filter_min_annotators.index)]
    train_min_data = train_min_data.iloc[:, :3]

    train_data_1 = train_data
    train_data = train_data[train_data['annotator'].isin(filter_max_annotators.index)]

    truth_data = pd.read_csv(
        '../data/music_true.csv',
        header=0,
        index_col=None,
        names=column_truth
    )

    # 获得真值，并按照task编号排序
    label_truth = truth_data[['Input.song', 'Input.true_label']]

    label_truth = label_truth.sort_values('Input.song', ascending=True)


    # 2.使用一个列表存放数据：[图片：399个工人]
    # 2.1 确定图片数 去掉重复的，并按照task排序
    images = train_data_1[column_images].drop_duplicates(column_images)
    images = images.sort_values('id', ascending=True)

    # 2.2 每张图片对应所有的工人 --找出所有工人
    annotators = train_data[['annotator']].drop_duplicates('annotator')
    annotators = annotators.reset_index(drop=True)

    annotators_one = annotators
    annotators = pd.merge(annotators_1, annotators, on='annotator', how='inner')
    annotators = annotators.drop(columns=['annotator'])

    # annotators = pd.get_dummies(annotators, columns=['annotator'])
    # 2.3 预测的数据
    predict_data_train = []
    for i in range(len(images)):
        new_images = pd.DataFrame(images.iloc[[i]].values.repeat(filter_max_annotators.shape[0], axis=0),
                                  columns=images.columns)
        new_images = new_images.reset_index(drop=True)
        predict_data_train.append(new_images)

    loaded_model = load_model('../train/musicModel_0.79')
    # 使用模型进行数据的填充
    results = []

    i = 0
    for result in predict_data_train:
        result = pd.concat([result, annotators_one], axis=1)

        result = result[column_images_swap]
        # print(result)

        result['id'] = result['id'].astype(int)

        result = result.drop(columns=['id', 'annotator'])
        y = loaded_model.predict([annotators, result])

        i = i + 1
        results.append(y)

    data_plus = []
    index = 0
    for result in results:

        y_pred = np.argmax(result, axis=1)
        y_pred = pd.DataFrame({'class': y_pred})

        y = pd.concat([annotators_one, y_pred], axis=1)

        x = []
        for i in range(filter_max_annotators.shape[0]):
            x.append(label_truth[['Input.song']].values[index][0])
        x = pd.DataFrame({'id': x})
        y = pd.concat([x, y], axis=1)
        index = index + 1
        data_plus.append(y)

    y_plus = pd.concat(data_plus, axis=0, ignore_index=True)
    pd.DataFrame(y_plus).to_csv('data/data-plus.csv', index=False)


    '''
    模型预测的与工人标注的结合在一起
    '''
    column_names = ['id', 'annotator', 'class']
    df1 = pd.read_csv('data/data-plus.csv')
    df1.rename(columns=dict(zip(df1.columns, column_names)), inplace=True)
    column_names = ['id', 'annotator', 'class']
    for i in range(0, 124):
        column_names.append("feature" + str(i))
    df2 = pd.read_csv(
        '../data/music_feature.csv',
        header=0,
        index_col=None,
        names=column_names
    )
    df2 = df2[['id', 'annotator', 'class']]
    df1 = df1.merge(df2[['id', 'annotator', 'class']], on=['id', 'annotator'], how='left', suffixes=('_x', ''))
    df1['class'] = df1['class'].fillna(df1['class_x'])
    df1 = df1.drop(['class_x'], axis=1)
    df1 = pd.concat([df1, train_min_data], axis=0)
    df1['class'] = df1['class'].astype(int)
    pd.DataFrame(df1).to_csv('data/data_fill_' + str(b) + '.csv', index=False)

if __name__ == '__main__':
    for b in range(0, 401, 10):
        print(b)
        G_model(b)
   #  # 一个图片对应399个工人
   #  # 1. 读取数据
   #  column_images = ['id']
   #  for i in range(0, 124):
   #      column_images.append("feature" + str(i))
   #
   #  column_images_swap = ['id', 'annotator']
   #  for i in range(0, 124):
   #      column_images_swap.append("feature" + str(i))
   #  column_truth = ['Input.song', 'Input.true_label']
   #  selected_columns = ['image_1', 'image_2', 'image_3', 'image_4', 'image_5']
   #  # train_data = pd.read_csv('data/filtered_df_20.csv', index_col=None)
   #  column_names = ['id', 'annotator', 'class']
   #  for i in range(0, 124):
   #      column_names.append("feature" + str(i))
   #  '''
   #  准备数据
   #  读取全部的数据，根据工人标注任务数进行数据的填充。
   #  '''
   #  train_data = pd.read_csv('../data/train/music_feature.csv', index_col=None)
   #  train_data = train_data[column_names]
   #  annotators = train_data[['annotator']].drop_duplicates('annotator')
   #  annotators = annotators.reset_index(drop=True)
   #
   #
   #  b = 250
   #
   #  annotator_counts = train_data['annotator'].value_counts()
   #  print(annotator_counts.values)
   #  filter_max_annotators = annotator_counts[annotator_counts > b]
   #  filter_min_annotators = annotator_counts[annotator_counts <= b]
   #  print(filter_min_annotators)
   #  annotators_1 = pd.get_dummies(annotators, columns=['annotator'])
   #  annotators_1 = pd.concat([annotators, annotators_1], axis=1)
   #
   #
   #  '''工人标注任务数大于n的数据，这些数据进行数据补全'''
   #  train_min_data = train_data[train_data['annotator'].isin(filter_min_annotators.index)]
   #  train_min_data = train_min_data.iloc[:, :3]
   #  print(train_min_data)
   #  train_data_1 = train_data
   #  train_data = train_data[train_data['annotator'].isin(filter_max_annotators.index)]
   #
   #
   #
   #  truth_data = pd.read_csv(
   #      '../data/train/music_true.csv',
   #      header=0,
   #      index_col=None,
   #      names=column_truth
   #  )
   #  print(truth_data)
   #  # 获得真值，并按照task编号排序
   #  label_truth = truth_data[['Input.song', 'Input.true_label']]
   #  print(label_truth)
   #  label_truth = label_truth.sort_values('Input.song', ascending=True)
   #  print(label_truth)
   #
   #  # 2.使用一个列表存放数据：[图片：399个工人]
   #  # 2.1 确定图片数 去掉重复的，并按照task排序
   #  images = train_data_1[column_images].drop_duplicates(column_images)
   #  images = images.sort_values('id', ascending=True)
   #  print(images)
   #  # 2.2 每张图片对应所有的工人 --找出所有工人
   #  annotators = train_data[['annotator']].drop_duplicates('annotator')
   #  annotators = annotators.reset_index(drop=True)
   #  print("annotator:", annotators)
   #  annotators_one = annotators
   #  annotators = pd.merge(annotators_1, annotators, on='annotator', how='inner')
   #  annotators = annotators.drop(columns=['annotator'])
   #  print(annotators)
   # # annotators = pd.get_dummies(annotators, columns=['annotator'])
   #  # 2.3 预测的数据
   #  predict_data_train = []
   #  for i in range(len(images)):
   #      new_images = pd.DataFrame(images.iloc[[i]].values.repeat(filter_max_annotators.shape[0], axis=0), columns=images.columns)
   #      new_images = new_images.reset_index(drop=True)
   #      predict_data_train.append(new_images)
   #
   #
   #  loaded_model = load_model('../train/musicModel_0.79')
   #  # 使用模型进行数据的填充
   #  results = []
   #  print(len(predict_data_train))
   #  i = 0
   #  for result in predict_data_train:
   #      result = pd.concat([result, annotators_one], axis=1)
   #
   #      result = result[column_images_swap]
   #      #print(result)
   #
   #      result['id'] = result['id'].astype(int)
   #      # result['annotator'] = result['annotator'].astype(int)
   #      result = result.drop(columns=['id', 'annotator'])
   #      y = loaded_model.predict([annotators, result])
   #      print(i)
   #      i = i+1
   #      results.append(y)
   #  # pd.DataFrame(results).to_csv('data/result_predict.csv')
   #  # 输出填充后的工人回答矩阵
   #  data_plus = []
   #  index = 0
   #  for result in results:
   #
   #      y_pred = np.argmax(result, axis=1)
   #      y_pred = pd.DataFrame({'class': y_pred})
   #
   #      y = pd.concat([annotators_one, y_pred], axis=1)
   #      #print(y)
   #      x = []
   #      for i in range(filter_max_annotators.shape[0]):
   #          x.append(label_truth[['Input.song']].values[index][0])
   #      x = pd.DataFrame({'id': x})
   #      y = pd.concat([x, y], axis=1)
   #      index = index + 1
   #      data_plus.append(y)
   #
   #
   #  y_plus = pd.concat(data_plus, axis=0, ignore_index=True)
   #  pd.DataFrame(y_plus).to_csv('data/data-plus.csv', index=False)
   #  #print(y_plus)
   #
   #  '''
   #  模型预测的与工人标注的结合在一起
   #  '''
   #  column_names = ['id', 'annotator', 'class']
   #  df1 = pd.read_csv('data/data-plus.csv')
   #  df1.rename(columns=dict(zip(df1.columns, column_names)), inplace=True)
   #  #df1 = df1.drop(columns=['id'])
   #  print(df1)
   #  column_names = ['id', 'annotator', 'class']
   #  for i in range(0, 124):
   #      column_names.append("feature" + str(i))
   #  df2 = pd.read_csv(
   #      '../data/train/music_feature.csv',
   #      header=0,
   #      index_col=None,
   #      names=column_names
   #  )
   #  print(df2)
   #  df2 = df2[['id', 'annotator', 'class']]
   #  df1 = df1.merge(df2[['id', 'annotator', 'class']], on=['id', 'annotator'], how='left', suffixes=('_x', ''))
   #  df1['class'] = df1['class'].fillna(df1['class_x'])
   #  df1 = df1.drop(['class_x'], axis=1)
   #  print(df1)
   #  print(filter_min_annotators)
   #  df1 = pd.concat([df1, train_min_data], axis=0)
   #  print(df1)
   #  df1['class'] = df1['class'].astype(int)
   #  pd.DataFrame(df1).to_csv('data/data_fill_' + str(b) + '.csv', index=False)
