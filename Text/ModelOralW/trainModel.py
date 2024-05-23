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
    train_data = pd.read_csv('../Data/textW2T.csv', index_col=None)
    print(train_data)
    column_names = ['task_id', 'annotator_id', 'answer']
    for i in range(0, 768):
        column_names.append(str(i))

    train_data = train_data[column_names]
    train_data = train_data.drop_duplicates(subset=['task_id'])
    id = train_data[['task_id']].astype(int).reset_index()
    train_label = train_data['answer']
    train_data_gate = train_data['annotator_id']
    train_data_gate = pd.get_dummies(train_data_gate, columns=['annotator_id'])
    train_data = train_data.drop(columns=['task_id', 'annotator_id', 'answer'])
    train_label = pd.get_dummies(train_label, columns=['answer'])

    return train_data_gate, train_data, train_label, id

def multiply(output1, output2):
    output = tf.multiply(output1, output2)
    output = K.sum(output, axis=2)
    return output

def accuracy(model, out_put, id):
    predict_result = model.layers[1].predict(out_put)

    predict_result = np.argmax(predict_result, axis=1)
    predict_result = pd.DataFrame({'label': predict_result})
    print(predict_result.shape)
    predict_result = pd.concat([id, predict_result], axis=1)
    print(predict_result)
    pd.DataFrame(predict_result).to_csv('../Data/bestWorkerPredict.csv')
    ground_truth = pd.read_csv('../data/text_truth.csv', index_col=0)

    predict_data = pd.read_csv('../Data/bestWorkerPredict.csv', index_col=0)
    print(predict_data)

    truth_dict = {}
    for index, data in ground_truth.iterrows():
        truth_dict[data['task_id']] = data['ground_truth']

    predict_dict = {}
    for index, data in predict_data.iterrows():
        predict_dict[data['task_id']] = data['label']

    count = 0

    for predict in predict_dict:
        if (predict_dict[predict] == truth_dict[predict]):
            count += 1

    print(count / len(predict_dict))
if __name__ == '__main__':

    gate_train_data, expert_train_data, answer_label_data, id = preparation_data()

    loaded_model = load_model('Train/model_BERT')
    gate_output = [0.0006728045890589879, 0.0010130616499509195, 0.003013659550367969, 0.0008860812434088566, 0.002766357440582161, 0.00071293266311831, 0.0006102063527172388, 0.000599193506921252, 0.0007340353583116031, 0.0006171194142817239, 0.003011521323933314,
                   0.0007982146620170378, 0.0008929267942715981, 0.00042775575136574077, 0.9832441296996933]

    expert_output = loaded_model.layers[2].predict(expert_train_data)
    print(expert_output)
    print(expert_output.shape)
    out_put = multiply(expert_output, gate_output)
    print(out_put)
    input_layer = Input(shape=out_put.shape[1:])
    new_output = loaded_model.layers[6](input_layer)
    new_model = Model(inputs=input_layer, outputs=new_output)
    # 编译新模型
    adam_optimizer = Adam(learning_rate=0.001)
    new_model.compile(loss=CategoricalCrossentropy(), optimizer=adam_optimizer, metrics=['accuracy'])
    new_model.summary()
    new_model.fit(
        x=out_put,
        y=answer_label_data,
        epochs=1000,
        batch_size=32
    )
    new_model.save("LimitedFModel_O")

    model = load_model("Train/LimitedFModel_O")
    accuracy(model, out_put, id)





