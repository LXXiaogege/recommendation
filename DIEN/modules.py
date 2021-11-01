from keras.layers import PReLU
from tensorflow.keras.layers import Layer, Dense, BatchNormalization
import tensorflow as tf

from DIEN.contrib.rnn_v2 import dynamic_rnn
from DIEN.contrib.utils import QAAttGRUCell, VecAttGRUCell


class AuxiliaryLoss(Layer):
    """
    辅助损失函数
    """

    def __init__(self, dnn_units=(100, 50)):
        super().__init__()

        # auxiliary_net, DNN
        self.bn = BatchNormalization(center=False, scale=False)
        self.dnn_layers = [Dense(units=units, activation='sigmoid') for units in dnn_units]
        self.final_dense = Dense(units=2, activation='softmax')

    def call(self, inputs):
        """
        noclick_seq: 不点击序列，negative sample
        h_states: 上一个时间步隐藏状态
        noclick_seq: click_seq: 点击序列， positive sample
        """
        h_states, click_seq, noclick_seq, mask = inputs

        click_input = tf.concat([click_seq, h_states], axis=-1)
        noclick_input = tf.concat([noclick_seq, h_states], axis=-1)

        for dense in self.dnn_layers:
            click_input = dense(click_input)

        for dense in self.dnn_layers:
            noclick_input = dense(noclick_input)

        click_prop = self.final_dense(click_input)[:, :, 0]
        noclick_prop = self.final_dense(noclick_input)[:, :, 0]

        click_loss_ = - tf.reshape(tf.math.log(click_prop),
                                   [-1, tf.shape(click_seq)[1]]) * mask  # tf.math.log(x) 计算出x的自然对数
        noclick_loss_ = - tf.reshape(tf.math.log(1.0 - noclick_prop), [-1, tf.shape(noclick_seq)[1]]) * mask

        loss_ = tf.reduce_mean(click_loss_ + noclick_loss_)
        return loss_


class DynamicGRU(Layer):
    def __init__(self, num_units=None, gru_type='GRU', return_sequence=True):
        """

        :param num_units:
        :param gru_type:
        :param return_sequence:
        """
        super().__init__()
        self.num_units = None
        self.gru_type = gru_type
        self.return_sequence = return_sequence

    def build(self, input_shape):
        # 创建一个可训练的权重变量
        input_seq_shape = input_shape[0]
        if self.num_units is None:
            self.num_units = input_seq_shape.as_list()[-1]  # 如果GRU的隐藏单元个数不指定，就取embedding维度
        if self.gru_type == 'AGRU':
            self.gru_cell = QAAttGRUCell(self.num_units)
        elif self.gru_type == 'AUGRU':
            self.gru_cell = VecAttGRUCell(self.num_units)
        else:
            self.gru_cell = tf.compat.v1.nn.rnn_cell.GRUCell(self.num_units)

        super(DynamicGRU, self).build(input_shape)

    def call(self, inputs):
        """
        :param concated_embeds_value: None * field_size * embedding_size
        :return: None*1
        """
        # 兴趣抽取层的运算
        if self.gru_type == "GRU" or self.gru_type == "AIGRU":
            rnn_input, sequence_length = inputs
            att_score = None
        else:  # 这个是兴趣进化层，这个中间会有个注意力机制
            rnn_input, sequence_length, att_score = inputs

        # rnn_input, sequence_length, att_score = inputs
        rnn_output, hidden_state = dynamic_rnn(self.gru_cell, inputs=rnn_input, att_scores=att_score,
                                               sequence_length=tf.squeeze(sequence_length),
                                               dtype=tf.float32)

        if not self.return_sequence:  # 只返回最后一个时间步的结果
            return hidden_state
        else:  # 返回所有时间步的结果
            return rnn_output


class Attention(Layer):
    """
    Activation Unit, sum pooling就是attention里的那个加权求和嘛，attention本身就是输出单个向量
    """

    def __init__(self, att_hidden_units):
        super().__init__()
        # 论文中的 a() 前向反馈网络
        self.att_dense = [Dense(units=unit, activation=PReLU()) for unit in att_hidden_units]
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

        """
        mask
        因为这些填充的位置，其实是没什么意义的，所以attention机制不应该把注意力放在这些位置上，需要进行一些处理。
        具体的做法是，把这些位置的值加上一个非常大的负数(负无穷)，这样的话，经过 softmax，这些位置的概率就会接近0！
        而我们的 padding mask 实际上是一个张量，每个值都是一个Boolean，值为 false 的地方就是我们要进行处理的地方。
        """
        # outputs (None,maxlen)
        outputs = tf.squeeze(input=outputs, axis=-1)
        paddings = tf.ones_like(input=outputs) * (
                -2 ** 32 + 1)  # tf.ones_like复制一个与input shape一样全为一的tensor,(-2 ** 32 + 1)为float32为能存储的最大数值
        # 把mask中为0的置为-2 ** 32 + 1，否则置为outputs中的对应元素
        outputs = tf.where(condition=tf.equal(mask, 0), x=paddings, y=outputs)

        # softmax，论文中说不实现，但代码中却实现了（无所谓不是DIN的重点）
        outputs = tf.nn.softmax(logits=outputs)
        outputs = tf.expand_dims(input=outputs, axis=1)
        # outputs = tf.matmul(outputs, v)
        # outputs = tf.squeeze(input=outputs, axis=1)  # (None,embed_dim*behavior_num)

        return outputs
