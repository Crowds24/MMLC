import os

import keras
import numpy as np
import pandas as pd
import model
from keras.callbacks import LambdaCallback
from keras.losses import CategoricalCrossentropy

from sklearn.metrics import f1_score
from sklearn.utils import shuffle
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf
import random
# 200 32 0.001

def preparation_data():
    # 1.列名
    train_data = pd.read_csv('../data/data_train.csv', index_col=None)

    train_gate_data = train_data[['worker']]
    train_gate_data = pd.get_dummies(train_gate_data, columns=['worker'])
    train_label = train_data[['label']]
    train_label = pd.get_dummies(train_label, columns=['label'])
    train_data = train_data.drop(columns=['taskId', 'worker', 'label'])

    return train_gate_data, train_data, train_label

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
    set_global_determinism(seed=SEED)
    gate_train_data, expert_train_data, type_label_data = preparation_data()
    gate_train_data, expert_train_data, type_label_data = shuffle(gate_train_data, expert_train_data, type_label_data)
    print(gate_train_data.shape)
    in_features_expert = expert_train_data.shape[1]
    in_features_gate = gate_train_data.shape[1]
    num_experts = 16
    gate_input_layer = Input(in_features_gate)
    expert_input_layer = Input(in_features_expert)

    # 门网络的输出
    gate_output = model.gate_model(in_features_gate, num_experts)(gate_input_layer)
    # gate_output = expert_gate_model_tf.gate_model(in_features_expert, num_experts)(expert_input_layer)

    # 初始化共享模型, 得到共享模型处理数据后的结果
    shareModel = model.ShareMoE(in_features_expert, num_experts)
    share_output = shareModel(expert_input_layer)

    # 共享模型与门网络加权
    out_put = model.multiply(share_output, gate_output)
    final_out = model.tower_output(out_put.shape[1])(out_put)


    # 填充数据
    model = Model(inputs=[gate_input_layer, expert_input_layer], outputs=final_out)
    model.summary()
    adam_optimizer = Adam(learning_rate=0.001)
    model.compile(loss=CategoricalCrossentropy(), optimizer=adam_optimizer, metrics=['accuracy'])

    model.fit(
        x=[gate_train_data, expert_train_data],
        y=type_label_data,
        epochs=150,
        batch_size=32,
        shuffle=False,
        workers=1
    )

    model.save("modelImage")







