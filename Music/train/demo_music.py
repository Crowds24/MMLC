import os

import expert_gate_model_music
import keras
import numpy as np
import pandas as pd
from keras.callbacks import LambdaCallback
from keras.losses import CategoricalCrossentropy
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf
import random



def preparation_data():
    # 1.列名
    data = pd.read_csv('../data/music_feature.csv', index_col=None)
    train_data = data.drop(columns=["id", "annotator", "class"])
    worker_data = data['annotator']
    worker_data = pd.get_dummies(worker_data, columns=['annotator'])
    label_data = data['class']
    label_data = pd.get_dummies(label_data, columns=['class'])

    return worker_data, train_data, label_data

SEED = 42
def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


def set_global_determinism(seed=SEED):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

if __name__ == '__main__':

    gate_train_data, expert_train_data, type_label_data = preparation_data()
    gate_train_data, expert_train_data, type_label_data = shuffle(gate_train_data, expert_train_data, type_label_data)
    in_features_expert = expert_train_data.shape[1]
    in_features_gate = gate_train_data.shape[1]
    num_experts = 10
    gate_input_layer = Input(in_features_gate)
    expert_input_layer = Input(in_features_expert)

    # 门网络的输出
    gate_output = expert_gate_model_music.gate_model(in_features_gate, num_experts)(gate_input_layer)
    # 初始化共享模型, 得到共享模型处理数据后的结果
    shareModel = expert_gate_model_music.ShareMoE(in_features_expert, num_experts)
    share_output = shareModel(expert_input_layer)

    # 共享模型与门网络加权
    out_put = expert_gate_model_music.multiply(share_output, gate_output)
    final_out = expert_gate_model_music.tower_output(out_put.shape[1])(out_put)


    # 填充数据
    model = Model(inputs=[gate_input_layer, expert_input_layer], outputs=final_out)
    model.summary()
    adam_optimizer = Adam(learning_rate=0.005)
    model.compile(loss=CategoricalCrossentropy(), optimizer=adam_optimizer, metrics=['accuracy'])

    model.fit(
        x=[gate_train_data, expert_train_data],
        y=type_label_data,
        epochs=15000,
        batch_size=2042,
        shuffle=False,
        workers=1
    )

    model.save("musicModel")







