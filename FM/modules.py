from tensorflow.keras.layers import Layer
import tensorflow as tf
from tensorflow.keras.regularizers import l2


class FM_layer(Layer):
    def __init__(self, feature_columns, latent_dim, w_reg=1e-6, v_reg=1e-6):
        super(FM_layer, self).__init__()
        self.sparse_features = feature_columns
        self.index_mapping = []
        self.feature_len = 0
        for i in self.sparse_features:
            self.index_mapping.append(self.feature_len)
            self.feature_len += i['feat_num']
        self.latent_dim = latent_dim
        self.w_reg = w_reg
        self.v_reg = v_reg

    def build(self, input_shape):
        # w0 : global bias
        self.w0 = self.add_weight(shape=(1,), initializer=tf.zeros_initializer(), trainable=True)
        # w 每个特征的权重
        self.w = self.add_weight(shape=(self.feature_len, 1), initializer=tf.random_normal_initializer(),
                                 regularizer=l2(self.w_reg), trainable=True)
        # v: 特征的隐含因素 原文叫 特征的latent_dim个factor，可以理解为特征分解成了latent_dim个子因素
        self.v = self.add_weight(shape=(self.feature_len, self.latent_dim), initializer=tf.random_normal_initializer(),
                                 regularizer=l2(self.v_reg), trainable=True)

    def call(self, inputs):
        """
        因为这里inputs是sparse features ，embedding_lookup就等价于原文中用weight乘ont-hot向量了。
        （原文的目的其实就是embedding_lookup）
        """
        inputs = inputs + tf.convert_to_tensor(self.index_mapping)  # 先把inputs做成id形式，把所有特征混到一起看做一个整体

        # first order
        first_order = self.w0 + tf.reduce_sum(tf.nn.embedding_lookup(self.w, inputs), axis=1)  # (batch_size, 1)
        # second order
        second_inputs = tf.nn.embedding_lookup(self.v, inputs)  # (batch_size, fields, embed_dim)
        square_sum = tf.square(tf.reduce_sum(second_inputs, axis=1, keepdims=True))  # (batch_size, 1, embed_dim)
        sum_square = tf.reduce_sum(tf.square(second_inputs), axis=1, keepdims=True)  # (batch_size, 1, embed_dim)
        second_order = 0.5 * tf.reduce_sum(square_sum - sum_square, axis=2)  # (batch_size, 1)
        # outputs
        outputs = first_order + second_order
        return outputs
