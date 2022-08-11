from operator import itemgetter

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import heapq


def get_dataset(file):
    """
    userId,movieId,rating,timestamp
    :param file:
    :return:
    """
    data = pd.read_csv(file, sep=",")
    train_data, test_data = train_test_split(data, test_size=0.2, shuffle=True)
    return train_data, test_data


def cal_user_sim(train_data, user_num):
    user_sim_matrix = np.zeros(shape=(user_num, user_num))

    # 共现矩阵
    grouped = train_data.groupby('movieId')
    for m1, group in grouped:
        user_list = group['userId']
        for i in user_list:
            for j in user_list:
                user_sim_matrix[i - 1][j - 1] += 1

    # 相似度矩阵
    for i in range(user_num):
        for j in range(user_num):
            user_sim_matrix[i][j] = cosine_similarity(user_sim_matrix[i].reshape((1, len(user_sim_matrix[i]))),
                                                      user_sim_matrix[j].reshape((1, len(user_sim_matrix[j]))))
    return user_sim_matrix


def recommend(train_data, user, user_sim_matrix, n, k):
    topN = heapq.nlargest(n, range(len(user_sim_matrix[user - 1])), user_sim_matrix[user - 1].take)  # topN用户的userid
    topN_sim = [user_sim_matrix[i][user-1] for i in topN]  # 与topN相似用户的相似度
    sims = 0
    for i in topN_sim:
        sims += i
    train_grouped = train_data.groupby('movieId')
    wr = {}
    movie_dict = {}
    for index, group in train_grouped:
        group = group.sort_values(by='userId')
        for i, sim in zip(topN, topN_sim):
            for row in group.iterrows():
                if row[1][0] == i:
                    if index not in wr:
                        wr.setdefault(index, 0)
                    wr[index] += sim * row[1][2]
        if index in wr.keys():
            movie_dict.setdefault(index, 0)
            movie_dict[index] = wr[index] / sims
    topK = sorted(movie_dict.items(), key=itemgetter(1), reverse=True)  # 推荐的topK电影
    if len(topK) > k:
        topK = topK[:k]
    movie_list = []
    for i in topK:
        movie_list.append(i[0])
    return movie_list


def evaluate(train_data, test_data, user_sim_matrix, n, k):
    hit = 0
    all_rec_num = 0
    all_test_num = 0

    test_grouped = test_data.groupby('userId')
    train_grouped = train_data.groupby('userId')
    for userId, group in train_grouped:
        if userId not in test_data['userId']:
            continue
        real_seq = test_grouped.get_group(userId)['movieId'].tolist()
        predict_seq = recommend(train_data, userId, user_sim_matrix, n, k)
        hit += len(list(set(real_seq) & set(predict_seq)))  # 交集
        all_rec_num += k
        all_test_num += len(real_seq)

    print("计算准确率，召回率")
    auc = hit / all_rec_num
    recall = hit / all_test_num
    # coverage =
    print("准确率：", auc, "召回率", recall)


if __name__ == '__main__':
    rating_file = r"D:\data\ml-latest-small\ratings.csv"
    user_num = 610
    n = 5  # topN ,获得n个相似用户
    k = 10  # 推荐电影数
    train_data, test_data = get_dataset(rating_file)
    user_sim_matrix = cal_user_sim(train_data, user_num)
    evaluate(train_data, test_data, user_sim_matrix, n, k)
