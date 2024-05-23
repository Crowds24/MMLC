import os

import Image.ModelOralW.Model.getOralWModel as getOralWModel
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.callbacks import LambdaCallback
from keras.losses import CategoricalCrossentropy
from keras.utils.vis_utils import plot_model
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.models import load_model
import tensorflow as tf
from keras import backend as K


def preparation_data():
    # 1.列名
    train_data = pd.read_csv('../../data/truth_train_data.csv', index_col=0)
    # column_names = ['taskId', 'worker', 'type']
    # for i in range(1, 8192):
    #     column_names.append("feature_" + str(i))
    #
    # train_data = train_data[column_names]
    train_data = train_data.drop_duplicates(subset=['taskId'])
    id = train_data[['taskId']].reset_index()
    train_label = train_data['label']
    train_data_gate = train_data['worker']
    train_data_gate = pd.get_dummies(train_data_gate, columns=['worker'])
    train_data = train_data.drop(columns=['taskId', 'worker', 'label'])
    train_label = pd.get_dummies(train_label, columns=['label'])

    return train_data_gate, train_data, train_label, id

def multiply(output1, output2):
    output = tf.multiply(output1, output2)
    output = K.sum(output, axis=2)
    return output


def accuracy(model, out_put, id):
    predict_result = model.layers[1].predict(out_put)

    predict_result = np.argmax(predict_result, axis=1)
    predict_result = pd.DataFrame({'label': predict_result})
    predict_result = pd.concat([id, predict_result], axis=1)
    pd.DataFrame(predict_result).to_csv('../data/bestWorkerPredict.csv')

    ground_truth = np.load('../data/labels_train.npy')
    ground_truth = pd.DataFrame(ground_truth)
    ground_truth.reset_index(level=0, inplace=True)
    ground_truth.rename(columns={'index': 'taskId'}, inplace=True)
    predict_data = pd.read_csv('../data/bestWorkerPredict.csv', index_col=0)

    truth_dict = {}
    for index, data in ground_truth.iterrows():
        truth_dict[data['taskId']] = data[0]

    predict_dict = {}
    for index, data in predict_data.iterrows():
        predict_dict[data['taskId']] = data['label']

    count = 0

    for predict in predict_dict:
        if (predict_dict[predict] == truth_dict[predict]):
            count += 1
    print(len(predict_dict))
    print(count / len(predict_dict))
if __name__ == '__main__':

    gate_train_data, expert_train_data, type_label_data, id = preparation_data()
    loaded_model = load_model('../train/modelImage_origin')
    gate_output = [0.15637665048183208, 0.13700922936029397, 0.10525455015991184, 0.11814092547321019,
                   0.13975985612883285, 0.1041856070355747, 0.15313458746019312, 0.08613859390015134]

    expert_output = loaded_model.layers[2].predict(expert_train_data)
    print(expert_output)
    print(expert_output.shape)
    out_put = multiply(expert_output, gate_output)
    print(out_put)
    input_layer = Input(shape=out_put.shape[1:])
    loaded_model.layers[6].trainable = False
    new_output = loaded_model.layers[6](input_layer)
    new_model = Model(inputs=input_layer, outputs=new_output)
    # 编译新模型
    adam_optimizer = Adam(learning_rate=0.001)
    new_model.compile(loss=CategoricalCrossentropy(), optimizer=adam_optimizer, metrics=['accuracy'])
    new_model.summary()
    new_model.fit(
        x=out_put,
        y=type_label_data,
        epochs=100,
        batch_size=32
    )
    new_model.save("LimitedFModel_O")

    model = load_model("LimitedFModel_O")
    accuracy(model, out_put, id)





