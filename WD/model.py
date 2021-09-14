from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Dense
from tensorflow.keras.regularizers import l2
from modules import Deep, Wide


class WideDeep(Model):
    def __init__(self, feature_columns, hidden_units_num, dnn_dropout, embed_reg=1e-6):
        """

        :param feature_columns: 特征信息
        :param hidden_units_num:  隐藏层个数
        :param dnn_dropout: DNN部分dropout
        """
        super(WideDeep, self).__init__()
        self.sparse_feature_columns = feature_columns

        # input_dim : vocabulary size,output_dim: embed_dim,
        # input_length ??????????????
        self.embed = {
            'embed' + str(i): Embedding(input_dim=feat['feat_num'], input_length=1, output_dim=feat['embed_dim'],
                                        embeddings_initializer='random_uniform', embeddings_regularizer=l2(embed_reg))
            for i, feat in enumerate(self.sparse_feature_columns)
        }

        self.index_mapping = []
        self.feature_length = 0
        for feat in self.sparse_feature_columns:
            self.index_mapping.append(self.feature_length)
            self.feature_length += feat['feat_num']

        self.dnn_network = Deep(hidden_units_num, dnn_dropout)
        self.linear = Wide(self.feature_length, embed_reg)

        self.final_dense = Dense(1, activation=None)

    def call(self, inputs, **kwargs):
        """
        实现网络向前传播，相当于 torch中的forward函数
        在调用fit函数
        :return:
        """
