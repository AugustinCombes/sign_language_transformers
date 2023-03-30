from keras_nlp.layers.transformer_encoder import TransformerEncoder
# from keras_nlp.layers import SinePositionEncoding
import tensorflow as tf
tfkl = tf.keras.layers

def get_model(
        hp, 
        input_shape=(10, (2*(21 + 40)))
        ):
    
    inputs = tf.keras.Input(input_shape, dtype=tf.float32)
    
    vector = tfkl.Bidirectional(tfkl.GRU(hp['gru1'], return_sequences=True))(inputs) #hp['gru1']=96 first test
    vector = tfkl.BatchNormalization()(vector)
    vector = tfkl.Activation('gelu')(vector)
    
    for _ in range(1):
        vector = TransformerEncoder(intermediate_dim=hp['ff_dim'], num_heads=hp['nhead'], dropout=hp['input_dropout'])(vector) #hp['input_dropout']=0.3 first test
        #hp['nhead']=12, hp['ff_dim']=160
    vector = tfkl.Bidirectional(tfkl.GRU(hp['gru2']))(vector) #hp['gru2']=96 first test
    vector = tfkl.Dropout(hp['output_dropout'])(vector) #hp['output_dropout']=0.2 first test

    output = tfkl.Dense(250, activation="softmax")(vector)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model