from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding
from config import Config
from modules import Attention
import tensorflow as tf


class RippleNet(Model):
    def __init__(self, n_entity, n_relation):
        super(RippleNet, self).__init__()
        self.config = Config
        self.n_hop = self.config.n_hop
        self.using_all_hops = self.config.using_all_hops
        self.entity_num = n_entity
        self.relation_num = n_relation
        self.entity_embedding = Embedding(input_dim=self.entity_num, output_dim=self.config.embed_dim,
                                          embeddings_initializer='random_uniform',
                                          embeddings_regularizer=l2(1e-4),
                                          input_length=1)
        self.relation_embedding = Embedding(input_dim=self.relation_num,
                                            output_dim=self.config.embed_dim * self.config.embed_dim,
                                            embeddings_initializer='random_uniform',
                                            embeddings_regularizer=l2(1e-4),
                                            input_length=1)
        self._key_address = Attention()
        self.transform_matrix = self._key_address.transform_matrix
        self.bce_loss = tf.keras.losses.BinaryCrossentropy()

    def call(self, inputs, training=None, mask=None):
        """
        memories_*: 最里层的元素是tensor
        """
        memories_h, memories_r, memories_t, items, labels = inputs

        memories_h = tf.convert_to_tensor(memories_h)
        memories_r = tf.convert_to_tensor(memories_r)
        memories_t = tf.convert_to_tensor(memories_t)
        # item embedding
        items_embed = self.entity_embedding(items)  # batch,embed_dim

        # ripple set embedding : n_hop,batch,n_memory,embed_dim
        h_emb_list = []
        r_emb_list = []
        t_emb_list = []
        for i in range(self.n_hop):
            h_emb_list.append(self.entity_embedding(memories_h[:, i, :]))
            # batch, n_memory, embed_dim, embed_dim
            r_emb_list.append(tf.reshape(self.relation_embedding(memories_r[:, i, :]),
                                         shape=[-1, self.config.n_memory, self.config.embed_dim,
                                                self.config.embed_dim]))
            t_emb_list.append(self.entity_embedding(memories_t[:, i, :]))

        o_list, item_embed = self._key_address([h_emb_list, r_emb_list, t_emb_list, items_embed])

        # predict
        y = o_list[-1]
        if self.using_all_hops:
            for i in range(self.n_hop - 1):
                y += o_list[i]
        # [batch_size]
        scores = tf.reduce_sum(item_embed * y, axis=1)
        scores = tf.sigmoid(scores)

        # compute loss
        loss = self.compute_loss(labels, scores, h_emb_list, r_emb_list, t_emb_list)
        return scores, loss

    def compute_loss(self, labels, scores, h_emb_list, r_emb_list, t_emb_list):
        base_loss = tf.reduce_mean(self.bce_loss(labels, scores))

        kge_loss = 0
        for hop in range(self.n_hop):
            h_expanded = tf.expand_dims(h_emb_list[hop], axis=2)
            t_expanded = tf.expand_dims(t_emb_list[hop], axis=3)
            hRt = tf.squeeze(tf.matmul(tf.matmul(h_expanded, r_emb_list[hop]), t_expanded))
            kge_loss += tf.reduce_mean(tf.sigmoid(hRt))
        kge_loss = -self.config.kge_weight * kge_loss

        l2_loss = 0
        for hop in range(self.n_hop):
            l2_loss += tf.reduce_mean(tf.reduce_sum(h_emb_list[hop] * h_emb_list[hop]))
            l2_loss += tf.reduce_mean(tf.reduce_sum(t_emb_list[hop] * t_emb_list[hop]))
            l2_loss += tf.reduce_mean(tf.reduce_sum(r_emb_list[hop] * r_emb_list[hop]))
            if self.config.item_update_mode == "replace nonlinear" or self.config.item_update_mode == "plus nonlinear":
                l2_loss += tf.nn.l2_loss(self.transform_matrix)
        l2_loss = self.config.l2_weight * l2_loss

        loss = base_loss + kge_loss + l2_loss
        return loss
