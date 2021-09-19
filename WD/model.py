import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Dense, Input
from tensorflow.keras.regularizers import l2
from .modules import Deep, Wide


class WideDeep(Model):
    def __init__(self, feature_columns, hidden_units_num, dnn_dropout=0., embed_reg=1e-6, w_reg=1e-6):
        """

        :param feature_columns: 特征信息
        :param hidden_units_num:  隐藏层个数
        :param dnn_dropout: DNN部分dropout
        :param embed_reg: embedding regularizer 参数初始化
        """
        super(WideDeep, self).__init__()
        self.sparse_feature_columns = feature_columns

        # input_dim : vocabulary size,output_dim: embed_dim,
        # input_length ??????????????
        self.embed = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_uniform',
                                         embeddings_regularizer=l2(embed_reg))
            for i, feat in enumerate(self.sparse_feature_columns)
        }

        self.index_mapping = []
        self.feature_length = 0
        for feat in self.sparse_feature_columns:
            self.index_mapping.append(self.feature_length)
            self.feature_length += feat['feat_num']

        self.dnn_network = Deep(hidden_units_num, dnn_dropout)
        self.linear = Wide(self.feature_length, w_reg=w_reg)

        self.final_dense = Dense(1, activation=None)

    def call(self, inputs, **kwargs):
        """
        实现网络向前传播，相当于 torch中的forward函数
        :param: inputs: 输入的特征   shape:(None,39)其实为(batch,feat_num)   type: tensor
        :return:
        """

        # self.embed['embed_' + str(i)](inputs[:, 1]) 得到的是每个特征的tensor表示，一共39个，把这些tensor拼接起来（特征拼接）因为每个特征dim为8
        # 39*8 = 312 最后得到 spares_embed shape : (None,312)
        sparse_embed = tf.concat([self.embed['embed_{}'.format(i)](inputs[:, i])
                                  for i in range(inputs.shape[1])], axis=-1)
        # sparse_embed = tf.concat([self.embed['embed_{}'.format(i)](inputs[:, i]) for i in range(inputs.shape[1])],
        #                          axis=-1)

        x = sparse_embed  # shape : (batch_size, field * embed_dim)

        # deep
        deep_outputs = self.dnn_network(x)
        deep_outputs = self.final_dense(deep_outputs)

        # wide ???
        wide_inputs = inputs + tf.convert_to_tensor(self.index_mapping)  # list to tensor
        # print("wide_inputs", wide_inputs)
        wide_output = self.linear(wide_inputs)

        # joint
        result = tf.nn.sigmoid(0.5 * wide_output + 0.5 * deep_outputs)  # 0.5为各自权重,0为偏置
        return result

    # def summary(self, **kwargs):
    #     sparse_inputs = Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
    #     Model(inputs=sparse_inputs, outputs=self.call(sparse_inputs)).summary()
