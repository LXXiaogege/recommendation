from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding
from modules import FM, Attention
import tensorflow as tf


class AFM(Model):
    def __init__(self, feature_column, k):
        super(AFM, self).__init__()
        self.sparse_feat_col = feature_column
        self.embedding_list = [
            Embedding(input_dim=feat['feat_num'], output_dim=feat['embed_dim'], embeddings_initializer='random_normal',
                      embeddings_regularizer=l2(1e-6), input_length=1) for feat in self.sparse_feat_col]
        self.feat_len = 0
        self.index_mapping = []
        for feat in self.sparse_feat_col:
            self.index_mapping.append(self.feat_len)
            self.feat_len += feat['feat_num']
        self.fm = FM(self.feat_len)

        self.attention = Attention(k)

        self.bias = self.add_weight(name='bias', shape=(1,), initializer=tf.zeros_initializer())

    def call(self, inputs, training=None, mask=None):
        # batch ,field_num
        sparse_feat = inputs
        embed_list = [self.embedding_list[i](sparse_feat[:, i]) for i in range(sparse_feat.shape[1])]

        sparse_feat = sparse_feat + tf.convert_to_tensor(self.index_mapping)
        # Pair-wise Interaction Layer
        first_order, second_order = self.fm((embed_list, sparse_feat))

        # Attention Layer
        attention_outputs = self.attention(second_order)

        y = tf.sigmoid(self.bias + first_order + attention_outputs)
        return y
