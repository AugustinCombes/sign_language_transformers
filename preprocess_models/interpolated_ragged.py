import tensorflow as tf
tfkl = tf.keras.layers
from utils import *

lipsUpperOuter = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
lipsLowerOuter = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
lipsUpperInner = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
lipsLowerInner = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
lips_idx = list(set(lipsUpperOuter + lipsLowerOuter + lipsUpperInner + lipsLowerInner))
lips_idx = tf.constant(lips_idx, dtype=tf.int32)

# Rq: possible d'avoir des timesteps hand incomplets ? Avec par exemple quelques landmarks de hand NaN sur ce timestep ? -> nanmean

class Preprocess(tf.keras.layers.Layer):
    def __init__(self, normalized_length):
        super(Preprocess, self).__init__()
        self.normalized_length = normalized_length

    def call(self, frames):
        frames = frames[:, :, :2] #drop z axis

        # Hands
        lh_frames, rh_frames = frames[:, 468:489], frames[:, 522:]

        lh_frames_x, lh_frames_y = lh_frames[:, :, 0], lh_frames[:, :, 1]
        rh_frames_x, rh_frames_y = rh_frames[:, :, 0], rh_frames[:, :, 1]
        lh_frames, rh_frames = tf.stack([lh_frames_x, 1-lh_frames_y], axis=-1), tf.stack([1-rh_frames_x, 1-rh_frames_y], axis=-1)

        hand = tf.stack([lh_frames, rh_frames], axis=0)
        hand = tf.where(tf.math.is_nan(hand), tf.zeros_like(tf.math.is_nan(hand), dtype=tf.float32), hand)
        hand = tf.reduce_sum(hand, axis=0)

        handsNanMask = tf.cast(tf.reduce_sum(hand, axis=[1, 2]), tf.bool) ## drops timestep having no hand data
        hand = tf.boolean_mask(hand, handsNanMask, axis=0)

        # Pose
        # pose = frames[:, 489:522]

        # Lips
        lips = tf.gather(frames, lips_idx, axis=1)
        lips = tf.boolean_mask(lips, handsNanMask, axis=0)
        lips = tf.where(tf.math.is_nan(lips), tf_nan_mean(lips), lips)
        lips = tf.where(tf.math.is_nan(lips), tf.zeros_like(lips), lips)

        # return tfkl.Flatten()(tf.concat([hand, lips], axis=1))

        # Time reduction ?
        raw_data = tf.concat([hand, lips], axis=1)
        source_length = tf.shape(raw_data)[0]
        normalized_idx = tf.linspace(0.0, tf.cast(source_length-1, tf.float32), self.normalized_length+1)
        # normalized_idx = tf.cast(tf.round(normalized_idx, tf.int32), tf.int32)
        normalized_idx = tf.cast(normalized_idx, tf.int32)

        sampled = list()
        for idx in range(self.normalized_length):
            start, end = normalized_idx[idx], normalized_idx[idx+1]
            # end = end + tf.cast(tf.equal(start, end), tf.int32) # avoid null length
            sample = raw_data[start:end]
            # sample = tf.concat([tf_nan_mean(sample), tf_nan_std(sample)], axis=1) #changer ordre ici
            sample = tf_nan_mean(sample)
            sampled.append(sample)
        sampled = tf.concat(sampled, axis=0)

        #tmp
        sampled = tfkl.Flatten()(sampled)

        sampled = tf.where(tf.math.is_nan(sampled), tf.zeros_like(tf.math.is_nan(sampled), dtype=tf.float32), sampled)
        sampledNanMask = tf.cast(tf.reduce_sum(sampled, axis=1), tf.bool)
        sampled = tf.boolean_mask(sampled, sampledNanMask, axis=0)
        return sampled