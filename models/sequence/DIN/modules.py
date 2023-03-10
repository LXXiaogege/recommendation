import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, BatchNormalization, PReLU


class Attention(Layer):
    """
    Activation Unit, sum pooling就是attention里的那个加权求和嘛，attention本身就是输出单个向量
    """

    def __init__(self, att_hidden_units, activation='prelu'):
        super().__init__()
        # 论文中的 a() 前向反馈网络
        self.att_dense = [Dense(units=unit, activation=PReLU()) for unit in att_hidden_units]
        self.att_final_dense = Dense(1)

    def call(self, inputs):
        """

        :param inputs:
        :return: attention value
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

        """
        mask
        因为这些填充的位置，其实是没什么意义的，所以attention机制不应该把注意力放在这些位置上，需要进行一些处理。
        具体的做法是，把这些位置的值加上一个非常大的负数(负无穷)，这样的话，经过 softmax，这些位置的概率就会接近0！
        而我们的 padding mask 实际上是一个张量，每个值都是一个Boolean，值为 false 的地方就是我们要进行处理的地方。
        """
        # outputs (None,maxlen) :attention score
        outputs = tf.squeeze(input=outputs, axis=-1)
        paddings = tf.ones_like(input=outputs) * (
                -2 ** 32 + 1)  # tf.ones_like复制一个与input shape一样全为一的tensor,(-2 ** 32 + 1)为float32为能存储的最大数值
        # 把mask中为0的置为-2 ** 32 + 1，否则置为outputs中的对应元素
        outputs = tf.where(condition=tf.equal(mask, 0), x=paddings, y=outputs)

        # softmax，attention score 归一化
        outputs = tf.nn.softmax(logits=outputs)
        outputs = tf.expand_dims(input=outputs, axis=1)

        # 对value加权求和,得到attention value
        outputs = tf.matmul(outputs, v)
        outputs = tf.squeeze(input=outputs, axis=1)  # (None,embed_dim*behavior_num)

        return outputs


class Dice(Layer):
    """
    Data Adaptive Activation Function

    解决问题： ICS(Internal Covariate Shift):训练数据在经过网络的每一层后其分布都发生了变化,一般解决方法有
    1. Normalization
    2. batch Norm
    3. 自适应激活函数
    Normalization,batch Norm是通过更改数据分布来适应激活函数，自适应激活函数是让激活函数去适应数据分布
    https://www.bilibili.com/video/BV1zK4y137Se?p=4
    """

    def __init__(self, ):
        super().__init__()
        self.bn = BatchNormalization(center=False, scale=False)
        self.alpha = self.add_weight(shape=(), dtype=tf.float32, name='alpha')

    def call(self, x):
        x_normed = self.bn(x)
        x_p = tf.sigmoid(x_normed)
        return self.alpha * (1.0 - x_p) * x + x_p * x
