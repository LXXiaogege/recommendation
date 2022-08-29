from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf


class AutoRec(Model):
    def __init__(self, num_units, num_out, dropout_rate=0.5):
        """

        :param num_units: 自定义，隐藏层大小
        :param num_out:  num_users/num_items
        :param dropout_rate:
        """
        super(AutoRec, self).__init__()
        self.encoder = Dense(units=num_units, activation='sigmoid', use_bias=True)
        self.decoder = Dense(units=num_out, use_bias=True)
        self.dropout = Dropout(rate=dropout_rate)

    def call(self, inputs, training=None, mask=None):
        """

        :param inputs: 所有用户对item i 的评分/该用户对所有items的评分
        :param mask:
        :param training: 是否在训练
        :return: 对所有用户对item i评分的预测分数/该用户对所有items的预测分数
        """
        hidden = self.dropout(self.encoder(inputs))
        output = self.decoder(hidden)
        if training:  # 只对用户投票过的电影计算loss
            return output * tf.cast(tf.sign(inputs), dtype=tf.float32)
        else:
            return output

# if __name__ == '__main__':
#     model = AutoRec(num_units=500, num_out=50)
#     a = np.random.randint(0,5,(100,50))
#     model.call(a)
