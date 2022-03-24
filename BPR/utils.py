import pandas as pd
import random
import numpy as np
from sklearn.model_selection import train_test_split

"""
movieLens ratings:
userId,movieId,rate,timeStamp
"""


def sparse_feature(feat, feat_num, embed_dim=4):
    return {'feat': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}


def dense_features():
    pass


def neg_sampling(user_id, pos_list, num_item):
    neg = random.randint(1, num_item)
    while neg in set(pos_list):
        neg = random.randint(1, num_item)
    return neg


def create_dataset(file_path, threshold=2, k=100):
    """
    train data : (user,item_i,item_j) , 与用户的关联度i>j
    :param file_path:
    :param threshold: 阈值，评价过且分数大于阈值，为正样本，未评价过且小于等于为负样本
    :param k: 测试集正负样本比， 一个正样本对应k个负样本
    :return:
    """
    data = pd.read_csv(file_path, sep='::', engine='python', names=['user', 'item', 'rate', 'time'])
    num_item = data['item'].max()
    num_user = data['user'].max()
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=2022)
    train_pos_sample = train_data[train_data['rate'] > threshold].sort_values(['user', 'time'], ascending=True)
    test_pos_sample = test_data[test_data['rate'] > threshold].sort_values(['user', 'time'], ascending=True)

    train = [[], [], []]
    test = [[], [], []]
    # train data
    for user, df in train_pos_sample[['user', 'item']].groupby('user'):
        user_pos_list = list(df['item'])
        user_pos_num = len(user_pos_list)
        train[0].extend([user for i in range(user_pos_num)])
        train[1].extend(user_pos_list)
        user_neg_list = [neg_sampling(user, user_pos_list, num_item) for i in range(user_pos_num)]
        train[2].extend(user_neg_list)
    # test data
    for user, df in test_pos_sample[['user', 'item']].groupby('user'):
        user_pos_list = list(df['item'])
        user_pos_num = len(user_pos_list)
        test[0].extend([user for i in range(user_pos_num)])
        test[1].extend(user_pos_list)
        user_neg_list = [[neg_sampling(user, user_pos_list, num_item) for j in range(k)] for i in range(user_pos_num)]
        test[2].extend(user_neg_list)
        break

    # shuffle ,同时打乱多个list
    temp = list(zip(train[0], train[1], train[2]))
    random.shuffle(temp)
    train[0], train[1], train[2] = zip(*temp)

    temp = list(zip(test[0], test[1], test[2]))
    random.shuffle(temp)
    test[0], test[1], test[2] = zip(*temp)

    train = [np.array(train[0]), np.array(train[1]), np.array(train[2])]
    test = [np.array((test[0])), np.array(test[1]), np.array(test[2])]
    feature_column = [sparse_feature(feat='user', feat_num=num_user),
                      sparse_feature(feat='item', feat_num=num_item)]

    return feature_column, train, test


create_dataset('D:/data/ml-1m/ratings.dat', threshold=2, k=100)
