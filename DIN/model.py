from keras.regularizers import l2
from tensorflow.keras import Model
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, BatchNormalization, Dropout
from .modules import Attention, Dice


class DIN(Model):
    def __init__(self, feature_columns, behavior_feature_list, att_hidden_units=(80, 40),
                 ffn_hidden_units=(80, 40), att_activation='prelu', ffn_activation='prelu', maxlen=40, dnn_dropout=0.,
                 embed_reg=1e-4):
        """
        DIN Model
        :param feature_columns:
        :param behavior_feature_list: ???
        :param att_hidden_units:
        :param ffn_hidden_units:
        :param att_activation:
        :param ffn_activation:
        :param maxlen:
        :param dnn_dropout:
        :param embed_reg:
        """
        super().__init__()
        self.maxlen = maxlen

        # dense_features一般直接连接dense层 , sparse_features进行embedding
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.other_sparse_len = len(self.sparse_feature_columns) - len(
            behavior_feature_list)  # user behavior之外不需要经过attention的sparse features
        self.dense_len = len(self.dense_feature_columns)
        self.behavior_len = len(behavior_feature_list)  # 1

        # history seq 嵌入层
        self.seq_embedding_layer = [Embedding(input_dim=feat['feat_num'],
                                              output_dim=feat['embed_dim'],
                                              embeddings_initializer='random_uniform',
                                              embeddings_regularizer=l2(embed_reg),
                                              input_length=1)
                                    for feat in self.sparse_feature_columns
                                    if feat not in behavior_feature_list
                                    ]
        # sparse embedding 嵌入层
        self.sparse_embedding_layer = [Embedding(input_dim=feat['feat_num'],
                                                 output_dim=feat['embed_dim'],
                                                 embeddings_initializer='random_uniform',
                                                 embeddings_regularizer=l2(embed_reg),
                                                 input_length=1)
                                       for feat in self.sparse_feature_columns
                                       if feat in behavior_feature_list]
        self.attention_layer = Attention(att_hidden_units=att_hidden_units, activation=att_activation)

        self.bn = BatchNormalization(trainable=True)

        self.ffn = [Dense(units=units, activation='prelu' if ffn_activation == 'prelu' else Dice())
                    for units in ffn_hidden_units]

        self.dropout = Dropout(dnn_dropout)
        self.dense_final = Dense(1)

    def call(self, inputs, training=None, mask=None):
        # dense_inputs :连续特征， sparse_inputs: 离散特征， seq_inputs:user behavior history， item_inputs: target
        # dense_inputs and sparse_inputs is empty.
        # seq_inputs (None, maxlen, len(hist[i]))
        # item_inputs (None,len(hist[i]))
        dense_inputs, sparse_inputs, seq_inputs, item_inputs = inputs

        # attention ---> mask, if the element of seq_inputs is equal 0, it must be filled in.
        # 因为在用pad_sequences填history seq时默认用0做的padding，所以只需要看第一个元素是否为0
        mask = tf.cast(tf.not_equal(seq_inputs[:, :, 0], 0),
                       dtype=tf.float32)  # (None, maxlen), cast()数据类型转换。bool转换为float类型，False->0.,True->1.

        # other
        other_info = dense_inputs
        for i in range(self.other_sparse_len):
            other_info = tf.concat([other_info, self.embed_sparse_layers[i](sparse_inputs[:, i])], axis=-1)

        # history seq embedding
        # seq_inputs[:, :, i] shape: (None,maxlen) ,embedding input：2D tensor with shape: (batch_size, input_length)
        # self.embed_seq_layers[i](seq_inputs[:, :, i]).shape： (None,maxlen,embed_dim)
        # seq_embed : (None,maxlen,embed_dim*behavior_num)
        seq_embed = tf.concat([self.seq_embedding_layer[i](seq_inputs[0, 0, i])
                               for i in range(self.behavior_len)], axis=-1)
        # target embedding
        # target_embed:(None,embed_dim*behavior_num)
        target_embed = tf.concat([self.seq_embedding_layer[i](item_inputs[0, i])
                                  for i in range(self.behavior_len)], axis=-1)
        # attention :query , key , value mask
        user_info = self.attention_layer(target_embed, seq_embed, seq_embed, mask)

        if self.dense_len > 0 or self.other_sparse_len > 0:
            info_all = tf.concat([user_info, target_embed, other_info], axis=-1)
        else:
            info_all = tf.concat([user_info, target_embed], axis=-1)

        info_all = self.bn(info_all)

        for fc in self.ffn:
            info_all = fc(info_all)

        info_all = self.dropout(info_all)
        outputs = tf.nn.sigmoid(self.dense_final(info_all))

        return outputs
