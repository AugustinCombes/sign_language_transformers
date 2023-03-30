from os.path import join as pjoin
from os.path import exists
import os
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import json

from utils import tf_get_features

try :
    BASE_DIR = "/kaggle/input/asl-signs/"
    df = pd.read_csv(pjoin(BASE_DIR, "train.csv"))
except :
    BASE_DIR = "asl-signs"
    df = pd.read_csv(pjoin(BASE_DIR, "train.csv"))

path2label = dict(zip(df.path, df.sign))
label2int = json.load(open(pjoin(BASE_DIR, "sign_to_prediction_index_map.json"), 'rb'))

# KFold
def get_KFold_dataset(preprocessing, path='full_cv.tfrecord', n_splits=3, group=True):
    X_ds = tf.data.Dataset.from_tensor_slices(
        BASE_DIR + "/" + df.path.values
        ).map(tf_get_features)
    y_ds = tf.data.Dataset.from_tensor_slices(
        df.sign.map(label2int).values.reshape(-1,1)
        )

    if group:
        # Perform stratisfied kfold split
        sgkf = StratifiedGroupKFold(n_splits=n_splits, random_state=42, shuffle=True)

        fold2id = dict()
        for fold_idx, (index_train, index_valid) in enumerate(sgkf.split(df.path, df.sign, df.participant_id)):
            fold2id[fold_idx] = np.unique(df.participant_id.values[index_valid])
            
        id2fold = dict()
        for k,v in fold2id.items():
            for vv in v:
                id2fold[vv] = k

        g_ds = tf.data.Dataset.from_tensor_slices(
            df.participant_id.map(id2fold).values.reshape(-1, 1)
        )
    
    else:
        df['participant_id2'] = df.participant_id.astype(str) + '_' + df.sign
        skf = StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)
        df["fold"] = 0
        for fold_idx, (index_train, index_valid) in enumerate(skf.split(df.path, df.participant_id2)):
            df.fold.iloc[index_valid] = fold_idx
        
        g_ds = tf.data.Dataset.from_tensor_slices(
            df.fold.values.reshape(-1, 1)
        )
    
    with tf.io.TFRecordWriter(path) as writer:
        for example in zip(X_ds.map(preprocessing), y_ds, g_ds):
            X, y, g = example
            feature = {
                'vector': tf.train.Feature(float_list=tf.train.FloatList(value=X.numpy().flatten())),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=y.numpy().flatten())),
                'group': tf.train.Feature(int64_list=tf.train.Int64List(value=g.numpy().flatten()))
            }
            example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example_proto.SerializeToString())

    def parse_function(example_proto):
        feature_description = {
            'vector': tf.io.FixedLenFeature(shape=(10*2*2*(21+21+40),), dtype=tf.float32),
            'label': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
            'group': tf.io.FixedLenFeature(shape=(), dtype=tf.int64)
        }
        parsed_example = tf.io.parse_single_example(example_proto, feature_description)
        return parsed_example['vector'], parsed_example['label'], parsed_example['group']

    ds = tf.data.TFRecordDataset(path)
    ds = ds.map(parse_function).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds