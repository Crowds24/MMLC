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

class LossAccHistory(keras.callbacks.Callback):
    def __init__(self):
        super(LossAccHistory, self).__init__()
        self.loss = []
        self.acc = []
        #self.val_loss = []
        #self.val_acc = []

    def on_epoch_end(self, epoch, logs={}):
        if logs.get('loss') > 4:
            self.loss.append(4)
        else:
            self.loss.append(logs.get('loss'))
        self.acc.append(logs['accuracy'])


class F1ScoreCallback(keras.callbacks.Callback):
    def __init__(self, gate_train_data, expert_train_data, class_label_data):
        super(F1ScoreCallback, self).__init__()
        self.gate_train_data = gate_train_data
        self.expert_train_data = expert_train_data
        self.class_label_data = class_label_data

    def on_epoch_end(self, epoch, logs={}):
        y_pred = np.argmax(self.model.predict([self.gate_train_data, self.expert_train_data]), axis=1)
        y_label = np.argmax(np.array(self.class_label_data), axis=1)
        f1 = f1_score(y_label, y_pred, average='macro')
        print(f'F1 score: {f1:.4f}')
        logs['val_f1'] = f1

def multiply(output1, output2):
    output = tf.multiply(output1, output2)
    output = K.sum(output, axis=2)
    return output

def preparation_data():
    # 1.列名
    train_data = pd.read_csv('../data/truth_train_data.csv', index_col=0)
    column_names = ['id', 'annotator', 'class']
    for i in range(0, 124):
        column_names.append("feature" + str(i))

    train_data = train_data[column_names]
    train_data = train_data.drop_duplicates(subset=['id'])
    print(train_data.shape)
    train_label = train_data['class']
    train_data_gate = train_data['annotator']
    train_data_gate = pd.get_dummies(train_data_gate, columns=['annotator'])
    id = train_data[['id']].astype(int).reset_index()
    print(id)
    train_data = train_data.drop(columns=['id', 'annotator', 'class'])
    train_label = pd.get_dummies(train_label, columns=['class'])

    return train_data_gate, train_data, train_label, id

def accuracy(model, out_put, id):

    predict_result = model.layers[1].predict(out_put)

    predict_result = np.argmax(predict_result, axis=1)
    predict_result = pd.DataFrame({'label': predict_result})
    print(predict_result.shape)
    predict_result = pd.concat([id, predict_result], axis=1)
    print(predict_result)
    pd.DataFrame(predict_result).to_csv('../data/bestWorkerPredict.csv')

    ground_truth = pd.read_csv('../data/music_true.csv')
    predict_data = pd.read_csv('../data/bestWorkerPredict.csv', index_col=0)
    print(predict_data)

    truth_dict = {}
    for index, data in ground_truth.iterrows():
        truth_dict[data['Input.song']] = data['Input.true_label']

    predict_dict = {}
    for index, data in predict_data.iterrows():
        predict_dict[data['id']] = data['label']

    count = 0

    for predict in predict_dict:
        if (predict_dict[predict] == truth_dict[predict]):
            count += 1

    print(count / len(predict_dict))

if __name__ == '__main__':

    gate_train_data, expert_train_data, class_label_data, id = preparation_data()
    loaded_model = load_model('../train/musicModel_0.79')
    gate_output = [0.13820427203489055, 0.17980818216228567, 0.10761719211584557, 0.038788771730331156, 0.029946872563708732, 0.2463344734515049,
                   0.08802976167845389, 0.06770221475833456, 0.09193783730082672, 0.01163042220381843]

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
        y=class_label_data,
        epochs=1500,
        batch_size=32
    )
    new_model.save("LimitedFModel_O")

    model = load_model("Train/LimitedFModel_O")
    accuracy(model, out_put, id)






