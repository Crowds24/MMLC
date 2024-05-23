import os
from random import random

import expert_gate_model_text
import keras
import numpy as np
import pandas as pd
from keras.callbacks import LambdaCallback
from keras.layers import Input
from keras.losses import CategoricalCrossentropy
from keras.models import Model
from keras.optimizers import Adam
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
import tensorflow as tf
import random

def preparation_data():
    # 1.列名
    data = pd.read_csv('../data/data_BERT.csv', index_col=None)
    train_data = data.drop(columns=["task_id", "annotator_id", "answer"])
    worker_data = data['annotator_id']
    worker_data = pd.get_dummies(worker_data, columns=['annotator_id'])
    label_data = data['answer']
    label_data = pd.get_dummies(label_data, columns=['answer'])

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
    # seed_value = 42
    # np.random.seed(seed_value)
    # tf.random.set_seed(seed_value)
    set_global_determinism(seed=SEED)
    gate_train_data, expert_train_data, type_label_data = preparation_data()
    gate_train_data, expert_train_data, type_label_data = shuffle(gate_train_data, expert_train_data, type_label_data)
    in_features_expert = expert_train_data.shape[1]
    in_features_gate = gate_train_data.shape[1]
    num_experts = 10
    gate_input_layer = Input(in_features_gate)
    expert_input_layer = Input(in_features_expert)

    # 门网络的输出
    gate_output = expert_gate_model_text.gate_model(in_features_gate, num_experts)(gate_input_layer)

    # 初始化共享模型, 得到共享模型处理数据后的结果
    shareModel = expert_gate_model_text.ShareMoE(in_features_expert, num_experts)
    share_output = shareModel(expert_input_layer)

    # 共享模型与门网络加权
    out_put = expert_gate_model_text.multiply(share_output, gate_output)
    final_out = expert_gate_model_text.tower_output(out_put.shape[1])(out_put)


    # 填充数据
    model = Model(inputs=[gate_input_layer, expert_input_layer], outputs=final_out)

    model.summary()
    adam_optimizer = Adam(learning_rate=0.001)
    model.compile(loss=CategoricalCrossentropy(), optimizer=adam_optimizer, metrics=['accuracy'])


    print(expert_train_data)
    print(gate_train_data)

    model.fit(
        x=[gate_train_data, expert_train_data],
        y=type_label_data,
        epochs=300,
        batch_size=2048,
        shuffle=False,
        workers=1
    )

    model.save("textModel")







