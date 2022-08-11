import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model


class BPR(Model):
    """
    目的：优化矩阵 user_embedding 和 item_embedding
    方法：通过加大 item_i 与 item_j 的差值来优化参数
    """

    def __init__(self, feat_column, embed_reg=1e-6):
        super(BPR, self).__init__()
        self.user_sparse_feat, self.item_sparse_feat = feat_column
        self.user_embedding = Embedding(input_dim=self.user_sparse_feat['feat_num'],
                                        output_dim=self.user_sparse_feat['embed_dim'],
                                        embeddings_initializer='random_normal',
                                        embeddings_regularizer=l2(embed_reg),
                                        input_length=1,
                                        mask_zero=False)
        self.item_embedding = Embedding(input_dim=self.user_sparse_feat['feat_num'],
                                        output_dim=self.user_sparse_feat['embed_dim'],
                                        embeddings_initializer='random_normal',
                                        embeddings_regularizer=l2(embed_reg),
                                        input_length=1,
                                        mask_zero=True)

        # ???
        self.mode = 'inner'

    def call(self, inputs, **kwargs):
        user, item_i, item_j = inputs
        user_embed = self.user_embedding(user)
        item_i_embed = self.item_embedding(item_i)
        item_j_embed = self.item_embedding(item_j)

        pos_scores = tf.reduce_sum(tf.multiply(user_embed, item_i_embed), axis=-1)
        neg_scores = tf.reduce_sum(tf.multiply(user_embed, item_j_embed), axis=-1)

        self.add_loss(tf.reduce_mean(-tf.math.log(tf.nn.sigmoid(pos_scores - neg_scores))))

        logits = tf.concat([pos_scores, neg_scores], axis=-1)
        return logits
