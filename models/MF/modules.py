import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import l2


class MF_layer(Layer):
    def __init__(self, user_num, item_num, latent_dim, use_bias=False, user_reg=1e-4, item_reg=1e-4,
                 user_bias_reg=1e-4, item_bias_reg=1e-4):
        super().__init__(self)
        self.user_num = user_num
        self.item_num = item_num
        self.latent_dim = latent_dim
        self.use_bias = use_bias
        self.user_reg = user_reg
        self.item_reg = item_reg
        self.user_bias_reg = user_bias_reg
        self.item_bias_reg = item_bias_reg

    def build(self, input_shape):
        # 矩阵分解后的用户矩阵
        self.p = self.add_weight(name='user_latent_matrix',
                                 shape=(self.user_num, self.latent_dim),
                                 initializer=tf.random_normal_initializer(),
                                 regularizer=l2(self.user_reg),
                                 trainable=True)  # trainable 是否作为模型参数参与训练
        # 矩阵分解后的物品矩阵
        self.q = self.add_weight(name='item_latent_matrix',
                                 shape=(self.item_num, self.latent_dim),
                                 initializer=tf.random_normal_initializer(),
                                 regularizer=l2(self.item_reg),
                                 trainable=True)

        # 为了消除用户和物品打分的偏差，引入用户和物品的偏差向量（因为每个用户的评分标准不一样，有人喜欢打高分，有人喜欢打低分）
        self.user_bias = self.add_weight(name='user_bias',
                                         shape=(self.user_num, 1),
                                         initializer=tf.random_normal_initializer(),
                                         regularizer=l2(self.user_bias_reg),
                                         trainable=self.use_bias)
        self.item_bias = self.add_weight(name='item_bias',
                                         shape=(self.item_num, 1),
                                         initializer=tf.random_normal_initializer(),
                                         regularizer=l2(self.item_bias_reg),
                                         trainable=self.use_bias)

    def call(self, inputs):
        """
        avg_score：全局偏差常数
        参考自王喆深度学习推荐系统书上的公式
        """
        # avg_score:(None,1), user_id: (None,1), item_id:(None,1)
        avg_score, user_id, item_id = inputs
        # 通过userid，itemid取出对应user、item的向量表示.  latent_user,latent_item: (None,1,latent_dim)
        latent_user = tf.nn.embedding_lookup(params=self.p, ids=user_id - 1)
        latent_item = tf.nn.embedding_lookup(params=self.q, ids=item_id - 1)

        outputs = tf.matmul(latent_item, latent_user, transpose_b=True)  # outputs:(None,1,1)
        if self.use_bias:
            user_bias = tf.nn.embedding_lookup(params=self.user_bias, ids=user_id - 1)
            item_bias = tf.nn.embedding_lookup(params=self.item_bias, ids=item_id - 1)
            outputs = tf.expand_dims(avg_score, axis=1) + user_bias + item_bias + outputs

        return outputs
