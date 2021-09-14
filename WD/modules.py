"""
network component
"""
from tensorflow.keras.layers import Layer, Dense, Dropout


class Wide(Layer):
    """
    Wide Component。 Linear part（线性模型）,  需要手动特征工程
    """

    def __init__(self):
        super(Wide, self).__init__()


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
