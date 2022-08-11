import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding
from models.DeepFM.modules import FM, Deep


class DeepFM(Model):

    def __init__(self, features_column, hidden_units):
        super().__init__()
        self.features_column_list = features_column
        self.features_embedding_list = [
            Embedding(input_dim=feat['feat_num'], output_dim=feat['embed_dim'], embeddings_initializer='uniform',
                      embeddings_regularizer=None, input_length=1) for feat in self.features_column_list]
        self.feat_len = 0
        self.index_mapping = []
        for feat in self.features_column_list:
            self.index_mapping.append(self.feat_len)
            self.feat_len += feat['feat_num']
        self.FM = FM(self.feat_len)
        self.Deep = Deep(hidden_units=hidden_units)

    def call(self, inputs, **kwargs):
        # batch_size,39
        sparse_features = inputs

        # embedding ,FM与Deep共享这一个embedding，Deep用了FM预训练好的Embedding向量
        sparse_features_embed = tf.concat(
            [self.features_embedding_list[i](sparse_features[:, i]) for i in range(sparse_features.shape[1])], axis=-1)

        # FM ， 数据经过FM得到的标量表示 y1.shape： batch，1
        sparse_features = sparse_features + tf.convert_to_tensor(self.index_mapping)
        y1 = self.FM((sparse_features,
                      tf.reshape(sparse_features_embed,
                                 shape=(-1, sparse_features.shape[1], self.features_column_list[0]['embed_dim']))))
        # Deep
        y2 = self.Deep(sparse_features_embed)  # batch，1

        outputs = tf.nn.sigmoid(tf.add(y1, y2))  # batch，1
        return outputs
