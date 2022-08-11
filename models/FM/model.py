from tensorflow.keras import Model
import tensorflow as tf
from models.FM import FM_layer


class FM(Model):
    def __init__(self, feature_columns, latent_dim, w_reg=1e-6, v_reg=1e-6):
        super().__init__(self)
        self.sparse_feature = feature_columns
        self.latent_dim = latent_dim
        self.w_reg = w_reg
        self.v_reg = v_reg
        self.fm_layer = FM_layer(feature_columns, self.latent_dim)

    def call(self, inputs, training=None, mask=None):
        outputs = self.fm_layer(inputs)
        outputs = tf.nn.sigmoid(outputs)
        return outputs

