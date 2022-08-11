import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Layer, Dense, Embedding, Dropout


class FM(Layer):
    def __init__(self, features_len):
        super().__init__()
        self.features_len = features_len

    def build(self, input_shape):
        # 1-order w, 一阶特征交互的权重，<w,x>.为每个特征分配一个权重
        self.w = self.add_weight(name='w', shape=(self.features_len, 1), initializer='random_normal',
                                 regularizer=l2(1e-6), trainable=True)

    def call(self, inputs, **kwargs):
        # embed_inputs: batch,39,embed_dim
        sparse_inputs, embed_inputs = inputs

        # first order： 给每个特征一个权重，加权求和，结果是一个标量， w与x内积
        first_order = tf.reduce_sum(tf.nn.embedding_lookup(params=self.w, ids=sparse_inputs), axis=1)
        # second order：特征两两相乘，赋予权重，加权求和, 这里的公式是FM论文里作者改写后的公式
        square_sum = tf.square(tf.reduce_sum(embed_inputs, axis=1, keepdims=True))
        sum_square = tf.reduce_sum(tf.square(embed_inputs), axis=1, keepdims=True)  # (batch_size, 1, embed_dim)
        second_order = 0.5 * tf.reduce_sum(square_sum - sum_square, axis=2)  # (batch_size, 1)
        # outputs = 偏差 + 原始特征 + 特征交叉后的特征
        outputs = first_order + second_order
        return outputs


class Deep(Layer):
    def __init__(self, hidden_units):
        super().__init__()
        self.dnn_network = [Dense(units=units, activation='relu') for units in hidden_units]
        self.dropout = Dropout(rate=0.2)
        self.final_dense = Dense(1)

    def call(self, inputs, **kwargs):
        x = inputs
        for dense in self.dnn_network:
            x = dense(x)
        outputs = self.final_dense(self.dropout(x))
        return outputs
