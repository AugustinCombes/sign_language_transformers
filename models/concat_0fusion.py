import tensorflow as tf
tfkl = tf.keras.layers
from keras_nlp.layers import SinePositionEncoding
from keras_nlp.layers.transformer_encoder import TransformerEncoder
from models.custom_layers import PrependClsTokenLayer, AddSinPosEncLayer

    
def get_tokenizer_concat(config):
    """
    Concatenate inputs modalities and tokenize with projection
    """

    modality_shapes = {m: config["modalities"][m]["raw_shape"] for m in config["modalities"].keys()}
    mod2inputs = {m: tfkl.Input(shape= (config["sequence_length"],) + mod_shape, name=m) 
                    for m, mod_shape in modality_shapes.items()}
    inputs = tf.concat(list(mod2inputs.values()), axis=-2)
    inputs = tfkl.TimeDistributed(tfkl.Flatten())(inputs)

    attention_mask = tf.math.reduce_any(
        tf.math.logical_not(tf.math.equal(inputs, 0)), axis=-1
            )
    attention_mask = tf.concat([tf.broadcast_to(tf.constant([True]), [tf.shape(attention_mask)[0], 1]), attention_mask], axis=1)
    
    multimodal_dim = sum([config["modalities"][m]["dim"] for m in config["modalities"].keys()])
    tokenization = tfkl.Dense(multimodal_dim, activation='relu')
    tokens = tokenization(inputs)

    tokens = AddSinPosEncLayer()(tokens)
    tokens = PrependClsTokenLayer()(tokens)

    return tf.keras.Model(
        inputs=list(mod2inputs.values()),
        outputs=[tokens, attention_mask]
    )

def get_concat_0fusion_model(config):
    modality_shapes = {m: config["modalities"][m]["raw_shape"] for m in config["modalities"].keys()}
    mod2inputs = {m: tfkl.Input(shape= (config["sequence_length"],) + mod_shape, name=m) 
                    for m, mod_shape in modality_shapes.items()}
    
    tokenizer = get_tokenizer_concat(config)
    tokens, mask = tokenizer(mod2inputs)

    for _ in range(config["vanilla_encoder_layers"]["num"]):
        encoder = TransformerEncoder(
                    intermediate_dim=config["vanilla_encoder_layers"]["ff_dim"], 
                    num_heads=config["vanilla_encoder_layers"]["num_heads"], 
                    dropout=config["vanilla_encoder_layers"]["dropout"]
                    )
        tokens = encoder(tokens, mask)

    cls = tokens[:, 0, :]
    y_pred = tfkl.Dense(250, activation='softmax')(cls)

    return tf.keras.Model(
        inputs=list(mod2inputs.values()),
        outputs=[y_pred]
    )