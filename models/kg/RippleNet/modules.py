from tensorflow.keras.layers import Layer, Dense
from config import Config
import tensorflow as tf


class Attention(Layer):
    """
    _key_addressing.
    n_hop个Attention层
    """

    def __init__(self):
        super(Attention, self).__init__()
        self.config = Config
        self.n_hop = self.config.n_hop
        self.item_update_mode = self.config.item_update_mode
        self.transform_matrix = Dense(self.config.embed_dim, use_bias=False)

    def call(self, inputs, *args, **kwargs):
        """
        query: item
        key： head * relation  做了知识图谱假设，假设： head * relation = tail
        value: tail
        """
        h_emb_list, r_emb_list, t_emb_list, item_embed = inputs

        o_list = []
        for hop in range(self.n_hop):
            # [batch_size, n_memory, dim, 1]
            h_expanded = tf.expand_dims(h_emb_list[hop], axis=3)

            # [batch_size, n_memory, dim]
            Rh = tf.squeeze(tf.matmul(r_emb_list[hop], h_expanded), axis=3)

            # [batch_size, dim, 1]
            v = tf.expand_dims(item_embed, axis=2)

            # [batch_size, n_memory]
            probs = tf.squeeze(tf.matmul(Rh, v), axis=2)

            # [batch_size, n_memory]
            probs_normalized = tf.nn.softmax(probs)

            # [batch_size, n_memory, 1]
            probs_expanded = tf.expand_dims(probs_normalized, axis=2)

            # [batch_size, dim]
            o = tf.reduce_sum(t_emb_list[hop] * probs_expanded, axis=1)

            item_embed = self.update_item_embedding(item_embed, o)
            o_list.append(o)
        return o_list, item_embed

    def update_item_embedding(self, item_embed, o):
        if self.item_update_mode == "replace":
            item_embed = o
        elif self.item_update_mode == "plus":
            item_embed = item_embed + o
        elif self.item_update_mode == "replace_transform":
            item_embed = self.transform_matrix(o)
        elif self.item_update_mode == "plus_transform":
            item_embed = self.transform_matrix(item_embed + o)
        else:
            raise Exception("Unknown item updating mode: " + self.item_update_mode)
        return item_embed
