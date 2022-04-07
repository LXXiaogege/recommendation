from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Layer, Dense, Dropout
import tensorflow as tf


class Linear(Layer):
    def __init__(self, feat_len):
        super(Linear, self).__init__()
        self.feat_len = feat_len

    def build(self, input_shape):
        self.w = self.add_weight(name='w', shape=(self.feat_len, 1), trainable=True, regularizer=l2(1e-6))

    def call(self, inputs, *args, **kwargs):
        row_sparse_feat = inputs
        # after embedding_lookup，before reduce sum: batch,39,1
        outputs = tf.reduce_sum(tf.nn.embedding_lookup(self.w, row_sparse_feat), axis=1)  # batch,1
        return outputs


class CIN(Layer):
    """
    Compressed Interaction Network
    局部上类似CNN，整体上类似RNN
    结构上，每一层的计算依赖于上一层的输出，类似于RNN
    计算公式上，类似于CNN
    """

    def __init__(self, cin_hidden_units):
        """
        cin_hidden_units: k个隐藏层单元数
        """
        super(CIN, self).__init__()
        self.cin_hidden_units = cin_hidden_units

    def build(self, input_shape):
        # get the number of embedding fields
        self.embedding_nums = input_shape[1]
        # a list of the number of CIN
        self.field_nums = [self.embedding_nums] + self.cin_hidden_units

        # filters
        # filter.shape : filter_width(kernel_size),in_channels,out_channels
        self.cin_W = {
            'CIN_W_' + str(i): self.add_weight(
                name='CIN_W_' + str(i),
                shape=(1, self.field_nums[0] * self.field_nums[i], self.field_nums[i + 1]),
                initializer='random_normal',
                regularizer=l2(1e-4),
                trainable=True)
            for i in range(len(self.field_nums) - 1)
        }

    def call(self, inputs, *args, **kwargs):
        # embed : batch,field_num,embed_dim
        embed = inputs
        x0 = embed
        dim = x0.shape[-1]
        hidden_layers = [x0]

        # 沿着dim维度做外积
        # split dimension 2 for convenient calculation, split_X_0 dtype: list
        split_X_0 = tf.split(hidden_layers[0], dim, 2)  # dim * (None, field_nums[0], 1)
        for idx, units in enumerate(self.cin_hidden_units):
            split_X_K = tf.split(hidden_layers[-1], dim, 2)  # dim * (None, filed_nums[i], 1)
            # 在每个embedding维度上做外积
            result_1 = tf.matmul(split_X_0, split_X_K,
                                 transpose_b=True)  # (dim, None, field_nums[0], field_nums[i])

            result_2 = tf.reshape(result_1, shape=[dim, -1, self.embedding_nums * self.field_nums[idx]])

            z = tf.transpose(result_2, perm=[1, 0, 2])  # (None, dim, field_nums[0] * field_nums[i])

            # 把z看做图片，做卷积，参数w为卷积核（过滤器）
            # 这里的filter与keras.layers.Conv1D的filter不同，这里的意思定义卷积参数，是tensor类型，
            # keras.layers.Conv1D中的filter，相当于pytorch里的out_channel,卷积核的个数，integer类型
            result_4 = tf.nn.conv1d(input=z, filters=self.cin_W['CIN_W_' + str(idx)], stride=1, padding='VALID')

            hidden_outputs = tf.transpose(result_4, perm=[0, 2, 1])  # (None, field_num[i+1], dim)

            hidden_layers.append(hidden_outputs)

        final_results = hidden_layers[1:]
        result = tf.concat(final_results, axis=1)  # (None, H_1 + ... + H_K, dim)
        result = tf.reduce_sum(result, axis=-1)  # (None, dim)

        return result


class Deep(Layer):
    def __init__(self, hidden_units):
        super(Deep, self).__init__()
        self.dnn = [Dense(units) for units in hidden_units]
        self.dropout = Dropout(0.2)

    def call(self, inputs, *args, **kwargs):
        x = inputs
        for dense in self.dnn:
            x = dense(x)
        outputs = self.dropout(x)
        return outputs
