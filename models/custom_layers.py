import tensorflow as tf
tfkl = tf.keras.layers
from keras_nlp.layers import SinePositionEncoding

class PrependClsTokenLayer(tfkl.Layer):
    def __init__(self, **kwargs):
        super(PrependClsTokenLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.cls_token = self.add_weight(shape=(1, 1, input_shape[-1]), initializer="random_normal", trainable=True, name='cls')
        super(PrependClsTokenLayer, self).build(input_shape)

    def call(self, x):
        batch_size = tf.shape(x)[0]
        cls_tokens = tf.broadcast_to(self.cls_token, [batch_size, 1, self.cls_token.shape[-1]])
        return tf.concat([cls_tokens, x], axis=1)
    
    
class AppendBottleneckTokensLayer(tfkl.Layer):
    def __init__(self, num_tokens, **kwargs):
        self.num_tokens = num_tokens
        super(AppendBottleneckTokensLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bottleneck_tokens = self.add_weight(shape=(1, self.num_tokens, input_shape["tokens"][-1]), initializer="random_normal", trainable=True, name='bottlenecks')
        super(AppendBottleneckTokensLayer, self).build(input_shape)

    @tf.function
    def call(self, tokens_with_mask):
        tokens, mask = tokens_with_mask["tokens"], tokens_with_mask["mask"]
        batch_size = tf.shape(tokens)[0]
        bottleneck_tokens = tf.broadcast_to(self.bottleneck_tokens, [batch_size, self.num_tokens, self.bottleneck_tokens.shape[-1]])

        add_mask = tf.broadcast_to(tf.constant([True]), [batch_size, self.num_tokens])
        mask = tf.concat([mask, add_mask], axis=1)

        return {
            "tokens": tf.concat([tokens, bottleneck_tokens], axis=1),
            "mask": mask
            }
    
class AddSinPosEncLayer(tfkl.Layer):
    def __init__(self, **kwargs):
        super(AddSinPosEncLayer, self).__init__(**kwargs)

    def call(self, x):
        return x + SinePositionEncoding()(x)