from keras_nlp.layers.transformer_encoder import TransformerEncoder
import tensorflow as tf
tfkl = tf.keras.layers

def get_model(hp):
    inputs = tf.keras.Input((10*2*21*2*2), dtype=tf.float32)
    vector = tfkl.Reshape((10, 2*21*2*2))(inputs)
    # vector = tfkl.Dense(32, activation='relu')(vector)
    vector = tfkl.Dense(hp["maindim"])(vector)
    vector = tfkl.BatchNormalization()(vector)
    vector = tfkl.Activation('gelu')(vector)
    # vector = tfkl.Dropout(0.2)(vector)
    for _ in range(hp["modules"]):
        vector = TransformerEncoder(intermediate_dim=hp["d_model"], num_heads=hp["heads"], dropout=hp["dropout"])(vector) #+= ?
    vector = tfkl.Bidirectional(tfkl.LSTM(hp["lstmdim"]))(vector)
    vector = tfkl.Dropout(hp["dropout2"])(vector)
    # vector = tfkl.GlobalMaxPooling1D(data_format="channels_last")(vector)

    output = tfkl.Dense(250, activation="softmax")(vector) #p-e logsoftmax plus stable ?
    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model