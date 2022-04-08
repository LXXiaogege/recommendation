from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Layer, Dense
import tensorflow as tf


class FM(Layer):
    def __init__(self, feat_len):
        super(FM, self).__init__()
        self.feat_len = feat_len

    def build(self, input_shape):
        self.w = self.add_weight(name='first_order_w', shape=(self.feat_len, 1),
                                 initializer=tf.random_normal_initializer(), regularizer=l2(1e-6), trainable=True)

    def call(self, inputs, *args, **kwargs):
        embed_list, sparse_feat = inputs
        # first order ,linear
        first_order = tf.reduce_sum(tf.nn.embedding_lookup(self.w, sparse_feat), axis=1)  # batch,1

        # pair-wise product
        second_order = []
        for i in range(len(embed_list)):
            for j in range(i + 1, len(embed_list)):
                second_order.append(embed_list[i] * embed_list[j])  # element-wise product
        # batch,[filed_num*(field_num-1)]/2,embed_dim
        second_order = tf.transpose(tf.convert_to_tensor(second_order), perm=[1, 0, 2])

        return first_order, second_order


class Attention(Layer):
    def __init__(self, k):
        super(Attention, self).__init__()
        self.att_dense = Dense(units=k, activation='relu', use_bias=True)
        self.att_dense_h = Dense(units=1)

        self.dense_p = Dense(units=1)

    def call(self, inputs, *args, **kwargs):
        # batch,[filed_num*(field_num-1)]/2,embed_dim
        second_order = inputs
        attention_weights = self.att_dense_h(self.att_dense(second_order))  # batch,[filed_num*(field_num-1)]/2,1
        # 每个二阶特征项对应一个weight
        attention_scores = tf.nn.softmax(attention_weights, axis=1)  # batch,[filed_num*(field_num-1)]/2,1
        outputs = tf.reduce_sum(second_order * attention_scores, axis=1)  # batch,embed_dim
        outputs = self.dense_p(outputs)  # batch,1
        return outputs
