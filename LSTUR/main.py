from tokenizers import Tokenizer
from torch import optim
from torch.utils.data import DataLoader
from LSTUR.Entity.NewsDataset import NewsDataset
from LSTUR.model.NewsEncoder import NewsEncoder, TitleEncoder

batch_size = 400
lr = 0.01

dropout_rate = 0.2
tokenizer = Tokenizer.from_file(r"D:\data\tokenizer/tokenizer-wiki.json")
vocab_size = tokenizer.get_vocab_size()  # 30000
embedding_dim = 200
kernel_size = 3
cnn_filter = 300

news_train = NewsDataset()
news_train_dataloader = DataLoader(news_train, batch_size=batch_size, shuffle=True, collate_fn=news_train.collate_fn)
titleEncoder = TitleEncoder(vocab_size, embedding_dim, cnn_filter, kernel_size, dropout_rate)
optimizer = optim.Adam(titleEncoder.parameters(), lr=lr)


