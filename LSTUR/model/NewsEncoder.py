from torch import nn
import torch


class TitleEncoder(nn.Module):
    """标题 Encoder"""

    def __init__(self, num_embeddings, attention_size, embedding_dim=200, out_channels=300, kernel_size=3,
                 dropout_rate=0.2):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings,
                                      embedding_dim=embedding_dim)  # output:[batch,seq_len,embed_dim]
        self.cnn = nn.Conv1d(in_channels=embedding_dim, out_channels=out_channels,
                             kernel_size=(kernel_size, kernel_size))  # output:[batch,out_channels,L]
        self.linear = nn.Linear(out_channels, attention_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        embedded = self.embedding(inputs)
        embedded = self.dropout(embedded)
        cnn = self.cnn(embedded)
        cnn = self.dropout(cnn)
        et = self.attention(cnn)
        return et

    def attention(self, ci):
        ai = torch.softmax(torch.tanh(self.linear(ci)), dim=-1)
        et = torch.sum(torch.matmul(ai, ci))
        return et


class NewsEncoder(nn.Module):
    """新闻 Encoder"""

    def __init__(self, num_embeddings, embedding_dim=200, out_channels=300, kernel_size=3, dropout_rate=0.2):
        super().__init__()
        self.titleEncoder = TitleEncoder(num_embeddings, attention_size=10, embedding_dim=200, out_channels=300,
                                         kernel_size=3, dropout_rate=0.2)

    def forward(self, title_list, topic_list, subtopic_list):
        title_present = self.titleEncoder(title_list)
        news_present = torch.cat(tensors=(subtopic_list, topic_list, title_present))
        return news_present
