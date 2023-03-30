import math
import time
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from os.path import join as pjoin

ROWS_PER_FRAME = 543  # number of landmarks per frame

def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)

def tf_get_features(ftensor):
    def feat_wrapper(ftensor):
        return load_relevant_data_subset(ftensor.numpy().decode('utf-8'))
    return tf.py_function(
        feat_wrapper,
        [ftensor],
        Tout=tf.float32
    )

def tf_nan_mean(x, axis=0, keepdims=True):
    return (tf.reduce_sum(
        tf.where(tf.math.is_nan(x), tf.zeros_like(x), x), 
        axis=axis, keepdims=keepdims) 
        / tf.reduce_sum(
            tf.where(
                tf.math.is_nan(x), tf.zeros_like(x), tf.ones_like(x)), 
            axis=axis, keepdims=keepdims))

def tf_nan_std(x, axis=0, keepdims=True):
    d = x - tf_nan_mean(x, axis=axis, keepdims=keepdims)
    return tf.math.sqrt(tf_nan_mean(d * d, axis=axis, keepdims=keepdims))

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)

def timeSpent(since):
    now = time.time()
    s = now - since
    return asMinutes(s)

class TimeLimitCallback(tf.keras.callbacks.Callback):
    def __init__(self, start_time, max_duration_hours=8, max_duration_minutes=30):
        super(TimeLimitCallback, self).__init__()
        self.start_time = start_time
        self.max_duration_seconds = max_duration_hours * 3600 + max_duration_minutes * 60

    def on_train_batch_end(self, batch, logs=None):
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.max_duration_seconds:
            self.model.stop_training = True
            print(f"Training stopped: time limit of {self.max_duration_seconds/3600:.1f} hours exceeded")

@tf.autograph.experimental.do_not_convert
def detuple(v, l, g, pid, sid):
    return (v, l)

@tf.autograph.experimental.do_not_convert
def ensure(shape):
    return lambda x : tf.ensure_shape(x, shape)

def scheduler(epoch, lr):
    if epoch < 8:
        return lr
    else:
        return lr * tf.math.exp(-0.1)
    
def create_load_tfrecords(new=False):
    idx = 0
    while f"run_{idx}" in os.listdir('.'):
        idx +=1

    if new:
        path = f"run_{idx}"
        print(f"Results will be saved at path {path}")
        os.makedirs(path)
        # !mkdir $path

    else:
        path = f"run_{idx-1}"
        print(f"Using path {path}")
    
    return pjoin(path, 'full_cv.tfrecord')