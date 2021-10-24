import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, BatchNormalization


class Attention(Layer):
    """
    Activation Unit, sum pooling就是attention里的那个加权求和嘛，attention本身就是输出单个向量
    """

    def __init__(self, att_hidden_units, activation='prelu'):
        super().__init__()
        # 论文中的 a() 前向反馈网络
        self.att_dense = [Dense(units=unit, activation=activation) for unit in att_hidden_units]
        self.att_final_dense = Dense(1)

    def call(self, inputs):
        """

        :param inputs:
        :return:
        """

        # query:candidate item  (None,embed_dim*behavior_num)
        # key : hist items (None,maxlen,embed_dim*behavior_num)
        # v: hist items (None,maxlen,embed_dim*behavior_num)
        # mask (None,seq_len)
        q, k, v, mask = inputs

        "把query处理成跟key一样的形状"
        # q (None,embed_dim*behavior_num*maxlen)
        q = tf.tile(input=q, multiples=[1, k.shape[1]])  # tile平铺，在指定维度上复制多次构成一个新tensor，multiples:每个维度复制次数
        # q (None,maxlen,embed_dim*behavior_num)
        q = tf.reshape(tensor=q, shape=([-1, k.shape[1], k.shape[2]]))

        # out product 原论文代码就是这样实现的
        # info (None , maxlen , embed_dim * behavior_num * 4 )
        info = tf.concat([q, k, q - k, q * k], axis=-1)

        # a(.) 前向反馈网络
        for dense in self.att_dense:
            info = dense(info)
        # outputs (None,maxlen,1)
        outputs = self.att_final_dense(info)

        "mask"
        # outputs (None,maxlen)
        outputs = tf.squeeze(input=outputs, axis=-1)
        paddings = tf.ones_like(input=outputs) * (
                -2 ** 32 + 1)  # tf.ones_like复制一个与input shape一样全为一的tensor,(-2 ** 32 + 1)为float32为能存储的最大数值
        # 把mask中为0的置为-2 ** 32 + 1，否则置为outputs中的对应元素
        outputs = tf.where(condition=tf.equal(mask, 0), x=paddings, y=outputs)

        # softmax，论文中说不实现，但代码中却实现了（无所谓不是DIN的重点）
        outputs = tf.nn.softmax(logits=outputs)
        outputs = tf.expand_dims(input=outputs, axis=1)
        outputs = tf.matmul(outputs, v)
        outputs = tf.squeeze(input=outputs, axis=1)  # (None,embed_dim*behavior_num)

        return outputs


class Dice(Layer):
    """
    Data Adaptive Activation Function
    当每层的输入有不同分布时，因为PRelu会不稳定，
    Dice可以被看作是PReLu的一般化。Dice的核心思想是根据输入数据的分布自适应地调整修正点，其值设为输入数据的均值。
    此外，Dice控制流畅，可在两个通道之间切换。当E(s) = 0, V ar[s] = 0时，Dice退化为PReLU
    """

    def __init__(self, ):
        super().__init__()
        self.bn = BatchNormalization(center=False, scale=False)
        self.alpha = self.add_weight(shape=(), dtype=tf.float32, name='alpha')

    def call(self, x):
        x_normed = self.bn(x)
        x_p = tf.sigmoid(x_normed)
        return self.alpha * (1.0 - x_p) * x + x_p * x
