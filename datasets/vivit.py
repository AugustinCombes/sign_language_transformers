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
def get_KFold_dataset(preprocessing, path='untitled', n_splits=3):
    dir_path = path
    path = pjoin(path, 'full_cv.tfrecord')

    def parse_function(example_proto):
        feature_description = {
            'pose': tf.io.FixedLenFeature(shape=(10, 5, 2), dtype=tf.float32),
            'hands': tf.io.FixedLenFeature(shape=(10, 21, 2), dtype=tf.float32),
            'eyes': tf.io.FixedLenFeature(shape=(10, 32, 2), dtype=tf.float32),
            'mouth': tf.io.FixedLenFeature(shape=(10, 20, 2), dtype=tf.float32),

            # 'pose_idx': tf.io.FixedLenFeature(shape=(10), dtype=tf.int64),
            # 'hands_idx': tf.io.FixedLenFeature(shape=(10), dtype=tf.int64),
            # 'eyes_idx': tf.io.FixedLenFeature(shape=(10), dtype=tf.int64),
            # 'mouth_idx': tf.io.FixedLenFeature(shape=(10), dtype=tf.int64),
            
            'label': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
            'group': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
            'pid': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
            'sid': tf.io.FixedLenFeature(shape=(), dtype=tf.int64)
        }
        parsed_example = tf.io.parse_single_example(example_proto, feature_description)
        return parsed_example, parsed_example['label']

    if exists(dir_path):
        print(f"Reloading dataset from path {path}")
    else:
        print(f"Dataset will be saved at path {path}")
        os.makedirs(dir_path)

        ds = tf.data.TFRecordDataset(path)
        X_ds = tf.data.Dataset.from_tensor_slices(
            BASE_DIR + "/" + df.path.values
            ).map(tf_get_features)
        y_ds = tf.data.Dataset.from_tensor_slices(
            df.sign.map(label2int).values.reshape(-1,1)
            )
        
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
        pid_ds = tf.data.Dataset.from_tensor_slices(
            df.participant_id.values
        )
        sid_ds = tf.data.Dataset.from_tensor_slices(
            df.sequence_id.values
        )
        
        with tf.io.TFRecordWriter(path) as writer:
            zipper = zip(X_ds.map(preprocessing), y_ds, g_ds, pid_ds, sid_ds)
            for example in tqdm(zipper):
                X, y, g, pid, sid = example
                # X, posX = X
                feature = {
                    'pose': tf.train.Feature(float_list=tf.train.FloatList(value=X["pose"].numpy().flatten())),
                    'hands': tf.train.Feature(float_list=tf.train.FloatList(value=X["hands"].numpy().flatten())),
                    'eyes': tf.train.Feature(float_list=tf.train.FloatList(value=X["eyes"].numpy().flatten())),
                    'mouth': tf.train.Feature(float_list=tf.train.FloatList(value=X["mouth"].numpy().flatten())),

                    # 'pose_idx': tf.train.Feature(int64_list=tf.train.Int64List(value=posX["pose"].numpy().flatten())),
                    # 'hands_idx': tf.train.Feature(int64_list=tf.train.Int64List(value=posX["hands"].numpy().flatten())),
                    # 'eyes_idx': tf.train.Feature(int64_list=tf.train.Int64List(value=posX["eyes"].numpy().flatten())),
                    # 'mouth_idx': tf.train.Feature(int64_list=tf.train.Int64List(value=posX["mouth"].numpy().flatten())),

                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=y.numpy().flatten())),
                    'group': tf.train.Feature(int64_list=tf.train.Int64List(value=g.numpy().flatten())),
                    'pid': tf.train.Feature(int64_list=tf.train.Int64List(value=pid.numpy().flatten())),
                    'sid': tf.train.Feature(int64_list=tf.train.Int64List(value=sid.numpy().flatten())),
                }
                example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example_proto.SerializeToString())

    ds = tf.data.TFRecordDataset(path)
    ds = ds.map(parse_function).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds