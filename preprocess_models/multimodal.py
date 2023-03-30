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
    def __init__(self):
        super(Preprocess, self).__init__()

    def call(self, frames):
        frames = frames[:, :, :2] #drop z axis

        # Hands
        lh_frames, rh_frames = frames[:, 468:489], frames[:, 522:]

        lh_frames_x, lh_frames_y = lh_frames[:, :, 0], lh_frames[:, :, 1]
        rh_frames_x, rh_frames_y = rh_frames[:, :, 0], rh_frames[:, :, 1]
        lh_frames, rh_frames = tf.stack([lh_frames_x, 1-lh_frames_y], axis=-1), tf.stack([1-rh_frames_x, 1-rh_frames_y], axis=-1)

        hand = tf.stack([lh_frames, rh_frames], axis=0)
        hand = tf.where(tf.math.is_nan(hand), tf.zeros_like(tf.math.is_nan(hand), dtype=tf.float32), hand)
        hand = tf.reduce_mean(hand, axis=0)

        handsNanMask = tf.cast(tf.reduce_sum(hand, axis=[1, 2]), tf.bool) ## drops timestep having no hand data
        hand = tf.boolean_mask(hand, handsNanMask, axis=0)

        # Pose
        # pose = frames[:, 489:522]

        # Lips
        # lips = tf.gather(frames, lips_idx, axis=1)

        # Time reduction ?


        #tmp
        hand = tfkl.Flatten()(hand)
        return hand
        lips_frames = tf.gather(frames, lips_idx, axis=1)

        features = tf.concat([lh_frames, rh_frames, lips_frames], axis=1)
        features = tf.where(tf.math.is_nan(features), tf.zeros_like(tf.math.is_nan(features), dtype=tf.float32), features)
        features = tfkl.Flatten()(features)
        return features
    
    #maybe faster processing if applied on ragged tensor directly ?