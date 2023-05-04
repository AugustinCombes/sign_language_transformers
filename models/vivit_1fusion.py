import tensorflow as tf
tfkl = tf.keras.layers
from keras_nlp.layers import SinePositionEncoding
from keras_nlp.layers.transformer_encoder import TransformerEncoder
from models.custom_layers import PrependClsTokenLayer, AddSinPosEncLayer

def get_tokenizer_vivit(config):
    """
    Tokenize each inputs modalities in same embedding space
    """
    modality_shapes = {m: config["modalities"][m]["raw_shape"] for m in config["modalities"].keys()}
    mod2inputs = {m: tfkl.Input(shape= (config["sequence_length"],) + mod_shape) 
                    for m, mod_shape in modality_shapes.items()}
    mod2flatten = {m: tfkl.TimeDistributed(tfkl.Flatten())(inputs) for m,inputs in mod2inputs.items()}
    
    attention_mask = tf.concat([tf.math.reduce_any(
        tf.math.logical_not(tf.math.equal(input, 0)), axis=-1)
            for input in mod2flatten.values()], axis=1)
    attention_mask = tf.concat([tf.broadcast_to(tf.constant([True]), [tf.shape(attention_mask)[0], 1]), attention_mask], axis=1)
    
    multimodal_dim = list(config["modalities"].values())[0]['dim']
    mod2tokenization = {m: tfkl.Dense(multimodal_dim, activation='relu', name=m) for m in config["modalities"].keys()}
    mod2tokens = {m: AddSinPosEncLayer()(
                    mod2tokenization[m](mod2flatten[m])
                    ) 
                    for m in config["modalities"].keys()}
    
    tokens = tf.concat(list(mod2tokens.values()), axis=-2)
    tokens = PrependClsTokenLayer()(tokens)

    return tf.keras.Model(
        inputs=[mod2inputs],
        outputs=[tokens, attention_mask]
    )

def get_vivit_1fusion_model(config):
    modality_shapes = {m: config["modalities"][m]["raw_shape"] for m in config["modalities"].keys()}
    mod2inputs = {m: tfkl.Input(shape= (config["sequence_length"],) + mod_shape, name=m) 
                    for m, mod_shape in modality_shapes.items()}
    
    tokenizer = get_tokenizer_vivit(config)
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