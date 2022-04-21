from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Conv2D, ZeroPadding2D, Dense, Flatten
from tensorflow.keras.initializers import RandomNormal
import tensorflow as tf
from modules import KMaxPooling
from tensorflow.keras.activations import relu, softmax


class CCPM(Model):

    def __init__(self, feat_column, conv_kernel_width=(6, 5, 3), conv_filters=(4, 4, 4), embed_reg=1e-6):
        super(CCPM, self).__init__()
        self.sparse_feat = feat_column
        self.sparse_feat_len = len(self.sparse_feat)
        self.conv_len = len(conv_filters)  # 卷积层数

        self.embedding_list = [
            Embedding(input_dim=feat['feat_num'], output_dim=feat['embed_dim'],
                      embeddings_initializer=RandomNormal(mean=0.0, stddev=0.0001, seed=2020),
                      embeddings_regularizer=l2(embed_reg), input_length=1)
            for feat in self.sparse_feat]

        # KMaxPooling
        self.p = []
        for i in range(1, self.conv_len + 1):
            if i < self.conv_len:
                k = max(1, int((1 - pow(i / self.conv_len, self.conv_len - i)) * self.sparse_feat_len))
                self.p.append(k)
            else:
                self.p.append(3)
        self.max_pooling_list = [KMaxPooling(k, dim=2) for k in self.p]

        self.padding_list = [ZeroPadding2D(padding=(0, conv_kernel_width[i] - 1))
                             for i in range(self.conv_len)]
        self.conv_list = [Conv2D(filters=conv_filters[i], kernel_size=(1, conv_kernel_width[i]))
                          for i in range(self.conv_len)]

        self.flatten = Flatten()
        self.dense = Dense(units=1)

    def call(self, inputs, training=None, mask=None):
        # batch,feat_num
        sparse_feat = inputs
        # batch,embed_dim,feat_num
        s = tf.stack([self.embedding_list[i](sparse_feat[:, i]) for i in range(self.sparse_feat_len)], axis=-1)
        # 先扩充channel维度
        s = tf.expand_dims(s, axis=3)

        for i in range(self.conv_len):
            # padding
            s = self.padding_list[i](s)
            # conv , batch,embed_dim,width,channel
            r = self.conv_list[i](s)
            s = self.max_pooling_list[i](r)
            s = relu(s)
        outputs = self.dense(self.flatten(s))
        outputs = softmax(outputs)
        return outputs
