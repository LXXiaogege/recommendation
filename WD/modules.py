"""
network component
"""
from tensorflow.keras.layers import Layer, Dense, Dropout
from tensorflow.keras.regularizers import l2
import tensorflow as tf


class Wide(Layer):
    """
    Wide Component。 Linear part（线性模型）,  需要手动特征工程
    """

    def __init__(self, feature_length, w_reg=1e-6):
        """

        :param feature_length: 特征长度
        :param w_reg:
        """
        super(Wide, self).__init__()
        self.feature_length = feature_length
        self.w_reg = w_reg

    def build(self, input_shape):
        """
        第一次调用call函数时，会在执行call函数前调用build函数
        :param input_shape:
        :return:
        """

        # 初始化权重weight
        self.w = self.add_weight(name='w', shape=(self.feature_length, 1),
                                 regularizer=l2(self.w_reg),
                                 trainable=True)

    def call(self, inputs, *args, **kwargs):
        # 为什么这样实现？？？？？？？
        # reduce_sum()计算tensor各维度之和
        result = tf.reduce_sum(tf.nn.embedding_lookup(self.w, inputs, ), axis=1)
        return result


class Deep(Layer):
    """
    Deep component , DNN part（前向反馈神经网络）
    只有全连接层组成
    """

    def __init__(self, hidden_units_num, dropout, activation='relu'):
        """

        :param hidden_units_num: 隐藏层个数
        :param dropout: dropout rate
        :param activation: 激活函数，论文中是 relu，修正线性单元
        """
        super(Deep, self).__init__()
        self.hidden_layers = [Dense(utils, activation=activation, use_bias=True) for utils in hidden_units_num]
        self.dropout = Dropout(dropout)

    def call(self, inputs, *args, **kwargs):
        x = inputs
        for h_layer in self.hidden_layers:
            x = h_layer(x)
            x = self.dropout(x)
        return x
