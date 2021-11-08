import math

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity


def get_dataset(file):
    """
    userId,movieId,rating,timestamp
    :param file:
    :return:
    """
    data = pd.read_csv(file, sep=",")
    train_data, test_data = train_test_split(data, test_size=0.2, shuffle=True)
    return train_data, test_data


def cal_movie_sim(train_data):
    """
    计算电影之间的相似度
    :return:dict item-item相似度矩阵
    """
    movie_popular = {}  # 统计训练集中都有哪些电影，且各自被看过多少次(原始数据中一个用户最多一次)
    movie_count = 0  # 训练集中电影数
    for index, row in train_data.iterrows():
        if row['movieId'] not in movie_popular:
            movie_popular[row['movieId']] = 0
        movie_popular[row['movieId']] += 1
    movie_count = len(movie_popular)

    # 如果用numpy直接建立数组movie id 最大值太大了,太耗内存了
    # movie_sim_matrix = np.zeros(shape=())  # 电影相似度矩阵
    movie_sim_matrix = {}
    grouped = train_data.groupby("userId")
    print("开始建立共现矩阵")
    for index, group in grouped:
        for m1 in group['movieId']:

            for m2 in group['movieId']:
                if m1 == m2:
                    continue
                movie_sim_matrix.setdefault(m1, {})
                movie_sim_matrix[m1].setdefault(m2, 0)
                # 同一个用户，遍历到其他用品则加1
                movie_sim_matrix[m1][m2] += 1
    print("item-item共现矩阵建立完成")

    # 计算电影之间的相似
    print("计算相似度矩阵")
    for m1, relate_movies in movie_sim_matrix.items():
        for m2, count in relate_movies.items():
            if movie_popular[m1] == 0 or movie_popular[m2] == 0:
                movie_sim_matrix[m1][m2] = 0
            else:
                # similarly = 共现次数/根号下各自出现次数相乘
                movie_sim_matrix[m1][m2] = count / math.sqrt(movie_popular[m1] * movie_popular[m2])
    print("相似度矩阵计算完成")
    return movie_sim_matrix


def evaluate(movie_sim_matrix, test_data):
    pass


if __name__ == '__main__':
    rating_file = r"D:\data\ml-latest-small\ratings.csv"
    # 候补电影数目
    cond_movies_num = 20
    # 要推荐给用户的数目
    rec_movies_num = 10

    train_data, test_data = get_dataset(rating_file)
    print(test_data)
    import sys
    sys.exit()
    movie_sim_matrix = cal_movie_sim(train_data)
    evaluate(movie_sim_matrix, test_data)