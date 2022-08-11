from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Dense
from tensorflow.keras.regularizers import l2
from modules import Linear, CIN, Deep
import tensorflow as tf


class xDeepFM(Model):
    def __init__(self, feat_column, hidden_units, cin_hidden_units):
        super(xDeepFM, self).__init__()
        self.sparse_feats = feat_column

        # params of linear layer
        self.feat_len = 0
        self.index_mapping = []
        for i in self.sparse_feats:
            self.index_mapping.append(self.feat_len)
            self.feat_len += i['feat_num']
        self.linear = Linear(self.feat_len)

        self.embedding_list = [
            Embedding(input_dim=feat['feat_num'], output_dim=feat['embed_dim'], embeddings_initializer='random_normal',
                      embeddings_regularizer=l2(1e-6), input_length=1) for feat in self.sparse_feats]
        # params of deep layer
        self.deep = Deep(hidden_units=hidden_units)
        self.last_dense = Dense(1)

        # params of CIN
        self.cin = CIN(cin_hidden_units)
        self.cin_dense = Dense(1)

        self.bias = self.add_weight(name='bias', shape=(1,), initializer=tf.zeros_initializer())

    def call(self, inputs, training=None, mask=None):
        sparse_feat = inputs

        embed_list = [self.embedding_list[i](sparse_feat[:, i]) for i in range(sparse_feat.shape[1])]
        embed = tf.transpose(tf.convert_to_tensor(embed_list), perm=[1, 0, 2])

        # linear layer
        linear_inputs = sparse_feat + tf.convert_to_tensor(self.index_mapping)
        linear_outputs = self.linear(linear_inputs)
        # deep layer
        # 初始化batch为None，reshape时shape不能指定batch
        deep_outputs = self.last_dense(self.deep(tf.reshape(tensor=embed, shape=(-1, embed.shape[1] * embed.shape[2]))))

        # CIN layer
        cin_outputs = self.cin_dense(self.cin(embed))

        # outputs
        outputs = tf.sigmoid(linear_outputs + deep_outputs + cin_outputs + self.bias)
        return outputs
