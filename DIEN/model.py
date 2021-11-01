from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, GRU, BatchNormalization, Dense, PReLU, Dropout, Permute
from tensorflow.keras.regularizers import l2

from DIEN.modules import AuxiliaryLoss, DynamicGRU, Attention
from DIN.modules import Dice
import tensorflow as tf


class DIEN(Model):
    """
    Interest Evolution Layer只实现了AUGRU
    """

    def __init__(self, feature_columns, behavior_columns, gru_type='gru',
                 use_negative_sample=False, alpha=1.0,
                 use_bn=False, dnn_hidden_units=(256, 128, 64), dnn_activation='relu',
                 att_hidden_units=(64, 16), att_activation="prelu", att_weight_normalization=True,
                 l2_reg_dnn=0, l2_reg_embedding=1e-6, dnn_dropout=0, seed=1024, task='binary'):
        """

        :param feature_columns: 所有特征列
        :param history_feature_list: user behavior sequence
        :param behavior_columns: 参与attention的特征列, user behavior features
        :param gru_type: gru类型，GRU AIGRU AUGRU AGRU
        :param use_negative_sample: bool,是否使用负采样
        :param alpha: float ,weight of auxiliary_loss
        :param use_bn: 是否使用Batch Normalization
        :param dnn_hidden_units:
        :param dnn_activation:
        :param att_hidden_units:
        :param att_activation:
        :param att_weight_normalization:
        :param l2_reg_dnn:
        :param l2_reg_embedding:
        :param dnn_dropout:
        :param seed:
        :param task:str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss  ???
        """
        super().__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.other_feature_len = len(self.sparse_feature_columns) - len(behavior_columns)
        self.dense_len = len(self.dense_feature_columns)
        self.behavior_len = len(behavior_columns)

        self.sparse_feature_embedding_layers = [Embedding(input_dim=feat['feat_num'],
                                                          output_dim=feat['embed_dim'],
                                                          embeddings_regularizer=l2(l2_reg_embedding),
                                                          input_length=1)
                                                for feat in self.sparse_feature_columns
                                                if feat in behavior_columns]

        # behavior layer
        self.seq_embedding_layers = [Embedding(input_dim=feat['feat_num'],
                                               output_dim=feat['embed_dim'], embeddings_initializer='uniform',
                                               embeddings_regularizer=l2(l2_reg_embedding), input_length=1)
                                     for feat in self.sparse_feature_columns
                                     if feat not in behavior_columns]

        # neg sample seq embedding layer
        self.neg_seq_embedding_layers = [Embedding(input_dim=feat['feat_num'],
                                                   output_dim=feat['embed_dim'], embeddings_initializer='uniform',
                                                   embeddings_regularizer=l2(l2_reg_embedding), input_length=1)
                                         for feat in self.sparse_feature_columns
                                         if feat not in behavior_columns]

        # interest extract layer
        self.gru = GRU(units=32, activation='sigmoid', return_sequences=True)

        self.alpha = alpha
        self.al = AuxiliaryLoss()

        # interest evolution layer
        self.attention_layer = Attention(att_hidden_units=att_hidden_units)

        # 后面的全连接跟DIN模型一样
        self.bn = BatchNormalization(trainable=True)

        self.ffn = [Dense(units=units, activation=PReLU() if dnn_activation == 'prelu' else Dice())
                    for units in dnn_hidden_units]

        self.dropout = Dropout(dnn_dropout)
        self.dense_final = Dense(1, activation='sigmoid')

    def call(self, inputs, training=None, mask=None):
        # dense_inputs and sparse_inputs is empty.
        # seq_inputs (None, maxlen, behavior_len)
        # item_inputs (None,behavior_len)
        dense_inputs, sparse_inputs, seq_inputs, no_click_seq, item_inputs, hist_real_len = inputs

        # other sparse feature embedding
        # sparse_embed = [self.sparse_feature_embedding_layers[i](sparse_inputs) for i in range(self.other_feature_len)]
        other_info = dense_inputs
        for i in range(self.other_feature_len):
            other_info = tf.concat([other_info, self.embed_sparse_layers[i](sparse_inputs[:, i])], axis=-1)

        # behavior embedding
        seq_embed = tf.concat([self.seq_embedding_layers[i](seq_inputs[:, :, i]) for i in range(self.behavior_len)],
                              axis=-1)

        # negative behavior sequence embedding
        no_click_embed = tf.concat(
            [self.neg_seq_embedding_layers[i](no_click_seq[:, :, i]) for i in range(self.behavior_len)], axis=-1)

        # target item embedding
        target_embed = tf.concat([self.seq_embedding_layers[i](item_inputs[:, i]) for i in range(self.behavior_len)],
                                 axis=-1)

        # seq_h = self.gru(seq_embed)
        embedding_size = None
        seq_h = DynamicGRU(embedding_size, return_sequence=True)(
            [seq_embed, hist_real_len])

        mask = tf.cast(tf.not_equal(seq_inputs[:, :, 0], 0), dtype=tf.float32)
        # seq_h[:, :-1, :]正样本，
        loss_ = self.al([seq_h[:, :-1, :], seq_embed[:, 1:, :], no_click_embed[:, 1:, :], mask[:, 1:]])

        scores = self.attention_layer([target_embed, seq_h, seq_h, mask])

        gru_type = 'AUGRU'
        final_state = DynamicGRU(embedding_size, gru_type=gru_type, return_sequence=False)(
            [seq_h, hist_real_len, Permute([2, 1])(scores)])

        if self.dense_len > 0 or self.other_feature_len > 0:
            outputs = tf.concat([final_state, target_embed, other_info], axis=-1)
        else:
            outputs = tf.concat([final_state, target_embed], axis=-1)

        outputs = self.bn(outputs)
        for dense in self.ffn:
            outputs = dense(outputs)
        outputs = self.dropout(outputs)
        outputs = self.dense_final(outputs)

        self.add_loss(self.alpha * loss_)
        return outputs
