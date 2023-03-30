import tensorflow as tf
from utils import *

lipsUpperOuter = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
lipsLowerOuter = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
lipsUpperInner = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
lipsLowerInner = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
lips_idx = list(set(lipsUpperOuter + lipsLowerOuter + lipsUpperInner + lipsLowerInner))
lips_idx = tf.constant(lips_idx, dtype=tf.int32)

class Preprocess(tf.keras.layers.Layer):
    def __init__(self, normalized_length):
        super(Preprocess, self).__init__()
        self.normalized_length = normalized_length

    def call(self, frames):
        frames = frames[:, :, :2]
        lh_frames = frames[:, 468:489]
        rh_frames = frames[:, 522:]
        lips_frames = tf.gather(frames, lips_idx, axis=1)

        raw_data = tf.concat([lh_frames, rh_frames, lips_frames], axis=1)
        source_length = tf.shape(raw_data)[0]
        normalized_idx = tf.linspace(0.0, tf.cast(source_length-1, tf.float32), self.normalized_length+1)
        normalized_idx = tf.cast(normalized_idx, tf.int32)

        sampled = list()
        for idx in range(self.normalized_length):
            start, end = normalized_idx[idx], normalized_idx[idx+1]
            sample = raw_data[start:end]
            sample = tf.concat([tf_nan_mean(sample), tf_nan_std(sample)], axis=1) #changer ordre ici
            sampled.append(sample)
        sampled = tf.concat(sampled, axis=0)
        features = tf.where(tf.math.is_nan(sampled), tf.zeros_like(sampled), sampled)

        diffs = tf.reshape(features, (self.normalized_length, 2*2*(#meanstd*dim
            2*21 + 40)))
        return diffs