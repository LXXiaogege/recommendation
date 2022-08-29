import os
import pandas as pd
from collections import defaultdict
import numpy as np


# auto encoder  rec data preprocess
def parse_ratings(root_path, sampling='random', mode='user-based'):
    """

    :param root_path: movieLens数据集根目录
    :param mode: 基于物品/用户 构建数据集,  user-based/item-based
    :param sampling: random / timestamp based
    :return:
    """
    path = os.path.join(root_path, 'ratings.dat')
    df = pd.read_csv(path, sep="::", names=['uid', 'mid', 'rating', 'timestamp'], engine='python')
    users_num = len(df['uid'].unique())  # 1 - 6040
    items_num = len(df['mid'].unique())  # 1 - 3952,有缺失，movie id不连续

    new2old_index = defaultdict(int)
    old2new_index = defaultdict(int)

    # 重置索引
    tem_idx = 0
    for idx, mid in enumerate(df['mid']):
        if mid not in old2new_index.keys():
            old2new_index[mid] = tem_idx
            new2old_index[tem_idx] = mid
            tem_idx += 1
        df['mid'].iloc[idx] = old2new_index[mid]
    df['uid'] = df['uid'].map(lambda x: x - 1)

    if mode == 'item-based':
        user2mid_seq = defaultdict(list)
        user2rating_seq = defaultdict(list)
        users_ratings_seq = []  # 所有用户的评分信息
        for user, value in df.groupby('uid'):
            value.sort_values(by='timestamp', ascending=True, inplace=True)
            user2mid_seq[user] = value['mid'].to_list()
            user2rating_seq[user] = value['rating'].to_list()
        # 缺失值补0
        for key, value in user2mid_seq.items():
            movies = value
            ratings = user2rating_seq[key]
            movies_rating = [0] * items_num  # 一个用户对所有items的评分
            for idx, i in enumerate(movies):
                movies_rating[i] = ratings[idx]
            users_ratings_seq.append(movies_rating)
        return np.array(users_ratings_seq), items_num
    else:
        mid2user_seq = defaultdict(list)
        mid2rating_seq = defaultdict(list)  # 每个电影对应所有用户的评分信息
        items_ratings_seq = []
        for movie, value in df.groupby('mid'):
            mid2user_seq[movie] = value['uid'].to_list()
            mid2rating_seq[movie] = value['rating'].to_list()
        for key, value in mid2user_seq.items():
            users = value
            ratings = mid2rating_seq[key]
            users_rating = [0] * users_num
            for idx, i in enumerate(users):  # user id 从1开始
                users_rating[i] = ratings[idx]
            items_ratings_seq.append(users_rating)
        return np.array(items_ratings_seq), users_num

# if __name__ == '__main__':
#     root_path = r"../data/ml-1m"
#     dataset = parse_ratings(root_path)
#     pass
