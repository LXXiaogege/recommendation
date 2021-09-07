from torch import nn
from LSTUR.model.NewsEncoder import NewsEncoder


class STUR(nn.Module):
    """用户兴趣短期表示"""

    def __init__(self, num_embeddings):
        super().__init__()
        self.newsEncoder = NewsEncoder(num_embeddings=num_embeddings)

    def forward(self, inputs):
        pass


class LTUR(nn.Module):
    """
    用户兴趣长期表示,对 User Id 嵌入
    """

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

    def forward(self, inputs):
        embedded = self.embedding(inputs)
        return embedded


class UserEncoderINI(nn.Module):
    """用户 Encoder,ini"""

    def __init__(self, uid_vocab_len, uid_embed_dim, news_vocab_len):
        super().__init__()
        self.lsur = LTUR(uid_vocab_len, uid_embed_dim)
        self.stur = STUR(news_vocab_len)

    def forward(self, inputs):

        pass
