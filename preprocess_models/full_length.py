import tensorflow as tf
tfkl = tf.keras.layers
from utils import *

lipsUpperOuter = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
lipsLowerOuter = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
lipsUpperInner = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
lipsLowerInner = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
lips_idx = list(set(lipsUpperOuter + lipsLowerOuter + lipsUpperInner + lipsLowerInner))
lips_idx = tf.constant(lips_idx, dtype=tf.int32)

class Preprocess(tf.keras.layers.Layer):
    def __init__(self):
        super(Preprocess, self).__init__()

    def call(self, frames):
        frames = frames[:, :, :2]
        lh_frames = frames[:, 468:489]
        rh_frames = frames[:, 522:]
        lips_frames = tf.gather(frames, lips_idx, axis=1)

        features = tf.concat([lh_frames, rh_frames, lips_frames], axis=1)
        features = tf.where(tf.math.is_nan(features), tf.zeros_like(tf.math.is_nan(features), dtype=tf.float32), features)
        features = tfkl.Flatten()(features)
        return features
    
    #maybe faster processing if applied on ragged tensor directly ?