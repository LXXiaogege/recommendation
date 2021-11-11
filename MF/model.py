from MF.modules import MF_layer
from tensorflow.keras import Model


class MF(Model):
    def __init__(self, features_column, use_bias=False):
        """
        use_bias ：是否使用用户，物品偏差
        """
        super().__init__()
        self.dense_features, self.sparse_features = features_column
        self.user_num, self.item_num = self.sparse_features[0]['feat_num'], self.sparse_features[1]['feat_num']
        self.latent_dim = self.sparse_features[0]['embed_dim']
        self.mf_layer = MF_layer(user_num=self.user_num, item_num=self.item_num, latent_dim=self.latent_dim,
                                 use_bias=use_bias)

    def call(self, inputs, training=None, mask=None):
        avg_score, user_id, item_id = inputs
        outputs = self.mf_layer([avg_score, user_id, item_id])
        return outputs
