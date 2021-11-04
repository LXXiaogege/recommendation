import collections
import math
import random


def get_center_and_context(dataset, max_window_size):
    """
    提取中心体和背景词
    :param dataset: 数据集
    :param max_window_size: 最大背景窗口
    :return: List 中心词，背景词,对应的索引为一对 中心词-背景词
    """
    centers, contexts = [], []
    for sentence in dataset:
        if len(sentence) < 2:  # 每个句子至少有两个词才能构成 中心词-背景词
            continue

        centers += sentence  # 每个词都可以作为中心词
        for center_i in range(len(sentence)):
            window_size = random.randint(1, max_window_size)  # 在整数1和max_window_size之间随机均匀采样一个整数作为背景窗口的大小
            indices = list(range(max(0, center_i - window_size),
                                 min(len(sentence), center_i + 1 + window_size)))
            indices.remove(center_i)
            contexts.append(indices)
    return centers, contexts


def get_negatives(all_contexts, sampling_weights, k):
    """
    负采样,正负样本 1:k, 例如一个center有2个positive context，有2*k个negative context，
    :param all_contexts: 背景词
    :param sampling_weights: 噪声词采样概率
    :param k: 噪声词（负样本）采样个数
    :return: all_negatives
    """
    all_negatives, neg_candidates, i = [], [], 0
    population = list(range(len(sampling_weights)))
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * k:
            if i == len(neg_candidates):
                # random.choices(population,weights,k)根据相对权重weights从population中选出k个元素
                i, neg_candidates = 0, random.choices(population=population, weights=sampling_weights,
                                                      k=int(1e5))  # 1e5:10的5次方
            neg, i = neg_candidates[i], i + 1
            if neg not in set(contexts):  # negative context 不能跟positive context一样
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives


def batchify(data):
    """
    小批量读取函数
    :param data:
    :return:
    """
    max_len = max(len(c) + len(n) for _, c, n in data)


if __name__ == '__main__':
    """
    数据预处理
    """
    data_path = r"D:\data\ptb"
    with open(data_path + r'\ptb.train.txt', 'r') as f:
        lines = f.readlines()
        raw_dataset = [sentence.split(sep=' ') for sentence in lines]

    "建立词索引,为计算简单只保留至少出现5次以上的词"
    counter = collections.Counter([tk for sentence in raw_dataset for tk in sentence])
    counter = dict(filter(lambda x: x[1] >= 5, counter.items()))
    # 把词映射到整数索引
    idx_to_token = [tk for tk, _ in counter.items()]  # 通过list index查询token
    token_to_idx = {tk: idx for idx, tk in enumerate(idx_to_token)}  # 通过dict token查询index

    # token dataset转为number dataset
    dataset = [[token_to_idx[token] for token in sentence if token in token_to_idx] for sentence in raw_dataset]
    num_tokens = sum([len(sentence) for sentence in dataset])

    "二次采样"


    def discard(idx):
        """
        越高频的词越容易被丢弃
        :param idx: 丢弃词的Index
        :return: Bool 是否要丢弃
        """
        return random.uniform(0, 1) < 1 - math.sqrt(1e-4 / counter[idx_to_token[idx]] * num_tokens)


    subsampled_dataset = [[token for token in sentence if not discard(token)] for sentence in dataset]

    "提取中心词和背景词"
    all_centers, all_contexts = get_center_and_context(dataset=subsampled_dataset, max_window_size=5)

    """
    负采样
    如果不经过负采样，训练没有负样本会造成预测结果全是1,
    k一般设为5，gensim也是5
    """
    sampling_weights = [counter[w] ** 0.75 for w in idx_to_token]  # word2vec论文建议负样本采样概率
    all_negatives = get_negatives(all_contexts=all_contexts, sampling_weights=sampling_weights, k=5)

    """
    读取数据集
    用随机小批量来读取 center和对应的context，neg_context
    """
