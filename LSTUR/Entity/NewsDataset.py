"""
新闻数据
"""
from tokenizers import Tokenizer
from torch.utils.data import Dataset
import pandas as pd
import torch

tokenizer = Tokenizer.from_file(r"D:\data\tokenizer/tokenizer-wiki.json")


class NewsDataset(Dataset):
    def __init__(self, path=r"D:\data\MIND Microsoft News Recommendation Dataset\MINDlarge_train\news.tsv"):
        df = pd.read_csv(path, sep='\t', header=None)
        self.ids = df[0]
        self.topics = df[1]
        self.subtopics = df[2]
        self.titles = df[3]

        self.n_topics = list(set(self.topics))
        self.num_topic = len(self.n_topics)  # class num :18

        self.n_subtopics = list(set(self.subtopics))
        self.num_subtopics = len(self.n_subtopics)  # class num :285

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return self.ids[index], self.topics[index], self.subtopics[index], self.titles[index]

    def topic_to_one_hot(self, topic):
        """把topic转为one-hot向量"""
        tensor = torch.zeros(self.num_topic)
        tensor[(self.n_topics.index(topic)) % self.num_topic] = 1
        return tensor

    def subtopic_to_one_hot(self, subtopic):
        """把subtopic转为one-hot向量"""
        tensor = torch.zeros(self.num_subtopics)
        tensor[(self.n_subtopics.index(subtopic)) % self.num_subtopics] = 1
        return tensor

    def collate_fn(self, batch):
        print("collate function......")
        title_list = []
        topic_list = []
        subtopic_list = []
        for idx, (id, topic, subtopic, title) in enumerate(batch):
            tokenizer.get_vocab()
            output = tokenizer.encode(title)
            topic_list.append(self.topic_to_one_hot(topic))
            subtopic_list.append(self.subtopic_to_one_hot(subtopic))
            title_list.append(output.ids)

        title_list = torch.LongTensor(title_list)
        topic_list = torch.LongTensor(topic_list)
        subtopic_list = torch.LongTensor(subtopic_list)
        return title_list, topic_list, subtopic_list
