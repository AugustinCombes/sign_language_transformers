import tensorflow as tf
tfkl = tf.keras.layers
from utils import *

import imageio
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

class Preprocess(tf.keras.layers.Layer):
    '''
    This is the first version of Preprocessing layer with visualisation of preprocessed data as gifs.
    - 4 modalities : pose, eyes, mouth, and normalized hand.
    - Each modality is individually time-interpolated according to interpolated_length, at the timesteps 
        having data for this modality.
    - A problem might happen for samples having less than 'interpolated_length' available timesteps.
    - Besides, it might be hard for the model to account for different modalities sampled at different
        timesteps.
    '''
    def __init__(self, interpolated_length=10, do_angles=True):
        super(Preprocess, self).__init__()
        self.fixed_frames = interpolated_length
        self.do_angles = do_angles

        self.mod2indexes = {
            # "pose": tf.constant([468, 500, 501, 502, 503, 522]), 
            "pose": tf.constant([504, 500, 501, 502, 503, 505]), 
            "hands": tf.constant([
                [468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488], 
                [522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542]
            ]), 
            "eyes": tf.constant([7, 33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246, 249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466]), 
            "mouth": tf.constant([13, 14, 78, 80, 81, 82, 87, 88, 95, 178, 191, 308, 310, 311, 312, 317, 318, 324, 402, 415])
        }

        self.lh_idx_range=(468, 489)
        self.rh_idx_range=(522, 543)

        self.mod2edge = {
            "pose": np.array([
                [4, 3, 1, 0, 2], #for right handed
                [4, 2, 1, 3, 0]  #for left handed
            ]),
            "hands": np.array([
                [0, 1, 2, 3, 4],
                [0, 5, 6, 7, 8],
                [0, 17, 18, 19, 20],
                [17, 13, 14, 15, 16],
                [5, 9, 10, 11, 12],
            ]), 
            "eyes": np.array([
                [1, 15, 12, 11, 10, 9, 8, 14, 2, 7, 6, 5, 4, 3, 13, 0, 1],
                [18, 30, 24, 25, 26, 27, 28, 31, 17, 16, 29, 19, 20, 21, 22, 23, 18]
            ]),
            "mouth": np.array([
                [2, 10, 3, 4, 5, 0, 14, 13, 12, 19, 11, 17, 16, 18, 15, 1, 6, 9, 7, 8, 2]
            ])
        }

        self.mod2style = {
            "pose": ('black', 3),
            "hands": ('blue', 2), 
            "eyes": ('black', 1),
            "mouth": ('red', 1)
        }
    
    @tf.function
    def pad_or_truncate_center(self, tensor, fixed_length):
        seq_length = tf.shape(tensor)[0]
        if seq_length > fixed_length:
            # Truncate the sequence by selecting the center patch
            start_idx = (seq_length - fixed_length) // 2
            tensor = tensor[start_idx:start_idx + fixed_length, :, :]
        elif seq_length < fixed_length:
            # Pad the sequence with zeros
            padding_length = fixed_length - seq_length
            padding = tf.zeros((padding_length, tf.shape(tensor)[1], tf.shape(tensor)[2]))
            tensor = tf.concat([tensor, padding], axis=0)
        return tensor

    def call(self, tensor):
        tensor = tensor[:, :, :2]  # drop z axis
        do_symetry = (tf.reduce_mean(tf.cast(tf.math.is_nan(tensor[:, self.rh_idx_range[0]:self.rh_idx_range[1], :]), tf.float32))
                    < tf.reduce_mean(tf.cast(tf.math.is_nan(tensor[:, self.lh_idx_range[0]:self.lh_idx_range[1], :]), tf.float32)))

        tensor = tf.cond(# symetry on 2nd axis when left-handed
            tf.equal(do_symetry, True),
            lambda: tf.stack([1 - tensor[:, :, 0], tensor[:, :, 1]], axis=-1),
            lambda: tensor
        )

        modalities, mod_indexes = {}, {}
        for modality, modalitidx in self.mod2indexes.items():
            mod_tensor = tf.gather(tensor, modalitidx, axis=1)
            
            if modality == "pose":
                mod_tensor = tf.cond(
                    do_symetry,
                    lambda: mod_tensor[:, 1:],
                    lambda: mod_tensor[:, :-1]
                )

            if modality == "hands":
                mod_tensor = tf.cond(
                    tf.equal(do_symetry, True),
                    lambda: mod_tensor[:, 1],
                    lambda: mod_tensor[:, 0]
                )
                mod_tensor = mod_tensor - mod_tensor[0]

            nan_frames_mask = tf.reduce_all(tf.math.is_nan(mod_tensor), axis=[1, 2])
            nan_frames_mask = tf.reshape(nan_frames_mask, [-1])
            
            mod_tensor = tf.boolean_mask(mod_tensor, tf.math.logical_not(nan_frames_mask), axis=0)
            mod_tensor = tf.where(tf.math.is_nan(mod_tensor), tf.zeros_like(mod_tensor), mod_tensor)
            mod_tensor = self.pad_or_truncate_center(mod_tensor, self.fixed_frames)

            if self.do_angles:
                mod_tensor = tf.math.atan2(mod_tensor[:, :, 0], mod_tensor[:, :, 1])

            modalities[modality] = mod_tensor
            # mod_indexes[modality] = mod_time_interpolated #time index

        return modalities
        # return modalities, mod_indexes
  
    def plot_sketch(self, array, modality):
        modalitidx = self.mod2edge[modality]
        col, width = self.mod2style[modality]
        for j, seg in enumerate(modalitidx):
            seg = np.array(seg)
            plt.plot(array[seg, 0], -array[seg, 1], '-', color=col, linewidth=width)    
        
    def output_gif_result(self, tensor, out_path='untitled.gif', gif_size=(224, 224)):

        #test if the signer is left_handed
        tensor = tensor[:, :, :2]  # drop z axis
        is_left_handed = (tf.reduce_mean(tf.cast(tf.math.is_nan(tensor[:, self.rh_idx_range[0]:self.rh_idx_range[1], :]), tf.float32))
                    < tf.reduce_mean(tf.cast(tf.math.is_nan(tensor[:, self.lh_idx_range[0]:self.lh_idx_range[1], :]), tf.float32))).numpy()
        
        mod2tensor = self.call(tensor)
        sequence_length = mod2tensor['hands'].shape[0]

        images = []
        for t in range(sequence_length):
            mod2tensor_t = {k:v[t].numpy() for k,v in mod2tensor.items()}
            # break
            fig, axes = plt.subplots(2, 2, figsize=(8, 8))
            axes = axes.flatten()

            for idx, (modality, array) in enumerate(mod2tensor_t.items()):
                
                modalitidx = self.mod2edge[modality]
                if modality == 'pose':
                    modalitidx = [self.mod2edge['pose'][1-int(is_left_handed)]]

                for j, seg in enumerate(modalitidx):
                    seg = np.array(seg)
                    axes[idx].plot(array[seg, 0], -array[seg, 1], '-')
                    axes[idx].plot(array[seg, 0], -array[seg, 1], 'yo')
                axes[idx].set_title(modality)

            # Save the figure to a buffer
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            im = Image.open(buf)
            images.append(im)
            plt.close()
        
        plt.suptitle('Test')

        # Save the images as a GIF file
        imageio.mimsave(out_path, images, fps=5)
        return out_path