from os.path import join as pjoin
import pandas as pd
import numpy as np
from tqdm import tqdm

import tensorflow as tf
tfkl = tf.keras.layers
from tqdm.keras import TqdmCallback
import json
import time
import math
import os

import warnings
warnings.simplefilter('ignore')

from preprocess_models.full_length import Preprocess
from models.LSTM import get_model
from utils import *


from utils import load_relevant_data_subset
from preprocess_models.multimodal_vivit import Preprocess
from datasets.vivit import get_KFold_dataset
from models.vivitlike import get_model

from kerastuner.tuners import BayesianOptimization
from keras import backend as backend

folds = 7
path = "preprocessed_datasets/vivit_experiment/"

ds = get_KFold_dataset(Preprocess(10), path)

#base parameters wihout search
hp = {
    'emb_dropout':0.,
    'dim_eyes':16,
    'dim_hands':16,
    'dim_mouth':16,
    'dim_pose':16,
    'dim_model':256,#177
    'num_enc':1,
    # 'ff_dim':1024,
    'nhead':8,
    't_dropout':0.4
}

def scheduler(epoch, lr):
    if epoch < 8 or epoch > 30:
        return lr
    else:
        return lr * tf.math.exp(-0.2) #note bien le 2

def get_model(hp):
    backend.clear_session()
    
    gru1 = hp.Int(name='gru1', min_value=16, max_value=208, step=64)
    nhead = hp.Int(name='nhead', min_value=4, max_value=16, step=4)
    ff_dim = hp.Int(name='ff_dim', min_value=64, max_value=256, step=64)
    input_dropout = hp.Float(name='input_dropout', min_value=0.1, max_value=0.5, step=0.1)
    gru2 = hp.Int(name='gru2', min_value=64, max_value=256, step=32)
    output_dropout = hp.Float(name='output_dropout', min_value=0.1, max_value=0.5, step=0.1)

    model = base_get_model({
        "gru1": gru1,
        'nhead': nhead,
        'ff_dim': ff_dim,
        'input_dropout': input_dropout,
        'gru2': gru2,
        'output_dropout': output_dropout,
    })

    lr = 1e-3
    model.compile(
        tf.keras.optimizers.Adam(learning_rate=lr),
        tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(),
        ]
    )
    return model

tuner = BayesianOptimization(
    get_model,
    objective='val_sparse_categorical_accuracy',
    max_trials=40,
    executions_per_trial=1,
    directory='',
    project_name=path,
)

val_accs = list()
fold_idx = 0

start = time.time()
train_ds = ds.filter(lambda v, l, g, pid, sid: g != fold_idx).map(detuple).padded_batch(32)
valid_ds = ds.filter(lambda v, l, g, pid, sid: g == fold_idx).map(detuple).padded_batch(32)

callbacks = [
        # TqdmCallback(verbose=0),
        tf.keras.callbacks.LearningRateScheduler(scheduler),
        tf.keras.callbacks.ModelCheckpoint(pjoin(path.split('/')[0], f"model_{fold_idx}"), 
            save_best_only=True, 
            save_weights_only=True,
            restore_best_weights=True, 
            monitor="val_sparse_categorical_accuracy", mode="max"),
        tf.keras.callbacks.EarlyStopping(patience=10, monitor="val_sparse_categorical_accuracy", mode="max", restore_best_weights=True)
        ]

tuner.search(
    train_ds, 
    epochs=50, 
    validation_data=valid_ds, 
    callbacks=callbacks
    )