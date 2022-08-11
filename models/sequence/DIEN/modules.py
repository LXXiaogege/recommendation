from tensorflow.keras.layers import PReLU
from tensorflow.keras.layers import Layer, Dense, BatchNormalization
import tensorflow as tf

from models.sequence.DIEN.contrib.rnn_v2 import dynamic_rnn
from models.sequence import QAAttGRUCell, VecAttGRUCell


class AuxiliaryLoss(Layer):
    """
    辅助损失函数（二分类）
    """

    def __init__(self, dnn_units=(100, 50)):
        super().__init__()

        # auxiliary_net, DNN
        self.bn = BatchNormalization(center=False, scale=False)

        # DNN代替内积操作
        self.dnn_layers = [Dense(units=units, activation='sigmoid') for units in dnn_units]
        self.final_dense = Dense(units=2, activation='softmax')

    def call(self, inputs):
        """
        h_states: 上一个时间步隐藏状态
        click_seq: 点击序列， positive sample sequence
        noclick_seq: 不点击序列，negative sample sequence
        """
        h_states, click_seq, noclick_seq, mask = inputs

        click_input = tf.concat([click_seq, h_states], axis=-1)
        noclick_input = tf.concat([noclick_seq, h_states], axis=-1)

        for dense in self.dnn_layers:
            click_input = dense(click_input)

        for dense in self.dnn_layers:
            noclick_input = dense(noclick_input)

        # 论文中的公式实现
        click_prop = self.final_dense(click_input)[:, :, 0]
        noclick_prop = self.final_dense(noclick_input)[:, :, 0]
        click_loss_ = - tf.reshape(tf.math.log(click_prop),
                                   [-1, tf.shape(click_seq)[1]]) * mask  # tf.math.log(x) 计算出x的自然对数
        noclick_loss_ = - tf.reshape(tf.math.log(1.0 - noclick_prop), [-1, tf.shape(noclick_seq)[1]]) * mask
        loss_ = tf.reduce_mean(click_loss_ + noclick_loss_)  # reduce_mean求均值

        return loss_


class DynamicGRU(Layer):
    """
    自定义GRU
    """

    def __init__(self, num_units=None, gru_type='GRU', return_sequence=True):
        """

        :param num_units: GRU隐藏单元个数
        :param gru_type: gru类型
        :param return_sequence: True返回整个hidden state sequence,False返回最后一个hidden state
        """
        super().__init__()
        self.num_units = num_units
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
    与DIN中的Attention相比不需要Value相乘。
    """

    def __init__(self, att_hidden_units):
        super().__init__()
        # 论文中的 a() 前向反馈网络
        self.att_dense = [Dense(units=unit, activation=PReLU()) for unit in att_hidden_units]
        self.att_final_dense = Dense(1)

    def call(self, inputs):
        """

        :param inputs:
        :return: attention score
        """
        # query:candidate item  (None,embed_dim*behavior_num)
        # key : hist items (None,maxlen,embed_dim*behavior_num)
        # v: hist items (None,maxlen,embed_dim*behavior_num)
        # mask (None,seq_len)
        q, k, v, mask = inputs

        # q (None,embed_dim*behavior_num*maxlen)
        q = tf.tile(input=q, multiples=[1, k.shape[1]])  # tile平铺，在指定维度上复制多次构成一个新tensor，multiples:每个维度复制次数
        # q (None,maxlen,embed_dim*behavior_num)
        q = tf.reshape(tensor=q, shape=([-1, k.shape[1], k.shape[2]]))

        # info (None , maxlen , embed_dim * behavior_num * 4 )
        info = tf.concat([q, k, q - k, q * k], axis=-1)

        for dense in self.att_dense:
            info = dense(info)
        # outputs (None,maxlen,1) 注意力得分
        outputs = self.att_final_dense(info)

        # outputs (None,maxlen)
        outputs = tf.squeeze(input=outputs, axis=-1)
        paddings = tf.ones_like(input=outputs) * (
                -2 ** 32 + 1)

        outputs = tf.where(condition=tf.equal(mask, 0), x=paddings, y=outputs)

        # 归一化
        outputs = tf.nn.softmax(logits=outputs)
        outputs = tf.expand_dims(input=outputs, axis=1)

        return outputs
