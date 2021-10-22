import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense


class Attention(Layer):
    def __init__(self, att_hidden_units, activation='prelu'):
        super().__init__()
        # 论文中的 a() 前向反馈网络
        self.att_dense = [Dense(units=unit, activation=activation) for unit in att_hidden_units]
        self.att_final_dense = Dense(1)

        def call(self, inputs):
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




