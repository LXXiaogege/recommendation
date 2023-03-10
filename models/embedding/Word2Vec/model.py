import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding


class Word2Vec(Model):
    """
    word2vec : skip-gram model
    """

    def __init__(self, vocab_size, embedding_dim):
        super().__init__()

        # center,context embedding也可以共用一个embedding layer

        self.center_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                                          embeddings_initializer='uniform', input_length=1)

        # input-length 为 负样本个数+1 (k+1)
        self.context_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=6)

    def call(self, inputs):
        # inputs : train_X
        # center (None,),  context: (None,max_len), mask:(None,max_len)
        center, context, mask = inputs
        center_embedded = self.center_embedding(center)
        context_embedded = self.context_embedding(context)

        # outputs； (None,1,max_len)
        outputs = tf.matmul(center_embedded, tf.transpose(context_embedded, perm=[0, 2, 1]))
        outputs = tf.squeeze(outputs, axis=1)  # (None,max_len)

        # mask
        paddings = tf.ones_like(input=outputs) * (-2 ** 32 + 1)
        outputs = tf.where(condition=tf.equal(mask, 0), x=paddings, y=outputs)

        outputs = tf.nn.sigmoid(outputs)

        return outputs
