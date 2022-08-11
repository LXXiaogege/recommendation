import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Dense,Dropout
from tensorflow.keras.regularizers import l2
from models.DeepCrossing.modules import ResidualUnits


class DeepCrossing(Model):
    def __init__(self, feature_column, hidden_units, res_dropout=0., embed_reg=1e-6):
        super(DeepCrossing, self).__init__()
        self.sparse_features = feature_column
        self.sparse_features_num = len(self.sparse_features)
        self.sparse_embedding_layer = [
            Embedding(input_dim=feat['feat_num'], output_dim=feat['embed_dim'], embeddings_initializer='random_uniform',
                      embeddings_regularizer=l2(embed_reg)) for feat in self.sparse_features]
        # dim stack， 经过stack层后的维度，为了保证每次输入到残差单元的维度相同
        dim_stack = sum([feat['embed_dim'] for feat in self.sparse_features])
        self.multiple_residual_units = [ResidualUnits(units, dim_stack=dim_stack) for units in hidden_units]
        self.dropout = Dropout(res_dropout)
        self.final_dense = Dense(1)

    def call(self, inputs, training=None, mask=None):
        sparse_inputs = inputs  # (batch,feature_column_num)
        # Embedding + Stack 层
        sparse_embed = tf.concat(
            [self.sparse_embedding_layer[i](sparse_inputs[:, i]) for i in range(self.sparse_features_num)],
            axis=-1)
        # Multiple Residual Units 层
        res = sparse_embed
        for residual_unit in self.multiple_residual_units:
            res = residual_unit(res)
        outputs = self.dropout(res)
        outputs = self.final_dense(outputs)
        # Score 层
        outputs = tf.nn.sigmoid(outputs)
        return outputs
