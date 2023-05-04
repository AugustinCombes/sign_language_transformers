from keras_nlp.layers.transformer_encoder import TransformerEncoder
from keras_nlp.layers import TokenAndPositionEmbedding
from einops import rearrange
import tensorflow as tf
tfkl = tf.keras.layers

def get_embedder_model(d_model, dropout, d_inter): 
    #may be better to optimize on the hidden dimension for each modality
    if d_inter==None:
        d_inter = 2*d_model

    return tf.keras.Sequential([
        tfkl.TimeDistributed(tfkl.Flatten()),
        tfkl.Dense(d_inter),
        tfkl.BatchNormalization(),
        tfkl.Dropout(dropout),
        tfkl.Activation('relu'),
        tfkl.Dense(d_model),
        tfkl.BatchNormalization(),
        tfkl.Dropout(dropout),
        tfkl.Activation('relu')
    ])

def get_model(hp):
    # 0 mask, 1 eyes, 2 hands, 3 mouth, 4 pose, 5 cls
    modalities = ['eyes', 'hands', 'mouth', 'pose']
    # modalities = ['hands', 'pose']

    mod2inputs = {
        'eyes': tfkl.Input(shape=(hp['seq_length'], 32, 2), name="eyes"), 
        'hands': tfkl.Input(shape=(hp['seq_length'], 21, 2), name="hands"), 
        'mouth': tfkl.Input(shape=(hp['seq_length'], 20, 2), name="mouth"), 
        'pose': tfkl.Input(shape=(hp['seq_length'], 5, 2), name="pose")
    }

    # attention mask
    attention_mask = tf.concat([tf.math.reduce_any(mod2inputs[mod] != 0, axis=[-2, -1]) for mod in modalities], axis=-1)
    attention_mask = tf.concat([tf.tile(tf.constant([True]), (tf.shape(attention_mask)[0],))[:, tf.newaxis], attention_mask], axis=-1)

    mod2model = {k:get_embedder_model(hp['dim_model'], hp['emb_dropout'], hp.get(f'dim_{k}', None)) for k in modalities}
    modalities_embedding = tf.stack([mod2model[mod](mod2inputs[mod]) for mod in modalities], axis=1)

    # mod-space & time encoding
    tokens = [k*tf.ones_like(mod2inputs[mod][:, :, 0, 0]) for k,mod in enumerate(modalities)]
    tokens = tf.stack(tokens, axis=1)

    tokens_embedder = TokenAndPositionEmbedding(6, hp['seq_length'], hp['dim_model'], mask_zero=True)
    tokens_embedding = tokens_embedder(tokens)

    # add to embedding
    modalities_ts = modalities_embedding + tokens_embedding
    modalities_ts = rearrange(modalities_ts, 'b m t d -> b (m t) d')

    cls = tokens_embedder(tf.constant([5]))
    cls = tf.tile(cls, (tf.shape(modalities_ts)[0], 1))[:, tf.newaxis]
    full_embedding = tf.concat([cls, modalities_ts], axis=1)

    for _ in range(hp['num_enc']):
        full_embedding = TransformerEncoder(intermediate_dim=hp['dim_model'], num_heads=hp['nhead'], dropout=hp['t_dropout'])(full_embedding, 
                                                                                                                              padding_mask=attention_mask)

    pool = full_embedding[:, 0]

    clas = tfkl.Dense(250, 'softmax')(pool)

    model = tf.keras.Model(inputs=list(mod2inputs.values()), outputs=clas)
    return model