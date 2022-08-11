from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Dense, Dropout,BatchNormalization
import tensorflow as tf
from models.NFM import Deep


class NFM(Model):
    def __init__(self, feat_column, hidden_units):
        super(NFM, self).__init__()
        self.feat_column = feat_column
        self.feat_len = 0
        for feat in feat_column:
            self.feat_len += feat['feat_num']

        self.embedding_list = [
            Embedding(input_dim=feat['feat_num'], output_dim=feat['embed_dim'], embeddings_initializer='uniform',
                      embeddings_regularizer=None, input_length=1) for feat in feat_column]
        self.dropout = Dropout(0.2)
        self.bn = BatchNormalization()

        self.dnn = Deep(hidden_units)
        self.last_hidden = Dense(1, activation='sigmoid', use_bias=False)

    def call(self, inputs, training=None, mask=None):
        # batch,filed_num
        sparse_feat = inputs
        # embedding layer
        embed_list = [self.embedding_list[i](sparse_feat[:, i]) for i in range(sparse_feat.shape[1])]
        embed = tf.stack(embed_list, axis=1)  # batch,field_num,embed_dim
        # B-interaction layer
        first_part = tf.square(tf.reduce_sum(embed, axis=-1))
        second_part = tf.reduce_sum(tf.square(embed), axis=-1)
        bi = self.dropout(0.5 * (first_part - second_part))
        bi = self.bn(bi)
        # hidden layers
        dnn_outputs = self.dnn(bi)
        # prediction score
        y = self.last_hidden(dnn_outputs)
        return y
