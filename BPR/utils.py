import pandas as pd
import random
import numpy as np
from tqdm import tqdm

"""
movieLens ratings:
userId,movieId,rate,timeStamp
"""


def sparse_feature(feat, feat_num, embed_dim=4):
    return {'feat': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}


def dense_features():
    pass


# 尽量少调用，否则效率很低
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
    print('数据预处理开始')
    data = pd.read_csv(file_path, sep='::', engine='python', names=['user', 'item', 'rate', 'time'])
    num_item = data['item'].max()
    num_user = data['user'].max()

    pos_sample = data[data['rate'] > threshold].sort_values(['user', 'time'], ascending=True)

    train = [[], [], []]
    val = [[], [], []]
    test = [[], [], []]
    print('构造数据集')
    # train data
    for user, df in tqdm(pos_sample[['user', 'item']].groupby('user')):
        user_pos_list = list(df['item'])
        user_pos_num = len(user_pos_list)
        if user_pos_num > 5:  # item太少不能构成训练集，验证集和测试集

            neg_list = [neg_sampling(user, user_pos_list, num_item) for i in range(user_pos_num + k)]

            train[0].extend(list(df['user'])[:-2])
            train[1].extend(user_pos_list[:-2])
            train[2].extend(neg_list[:-2 - k])

            val[0].append(user)
            val[1].append(user_pos_list[-2])
            val[2].append(neg_list[-2 - k])

            test[0].append(user)
            test[1].append((user_pos_list[-1]))
            test[2].extend([neg_list[user_pos_num:]])
    print('数据集准备完成')
    # shuffle ,同时打乱多个list
    # temp = list(zip(train[0], train[1], train[2]))
    # random.shuffle(temp)
    # train[0], train[1], train[2] = zip(*temp)
    #
    # temp = list(zip(val[0], val[1], val[2]))
    # random.shuffle(temp)
    # val[0], val[1], val[2] = zip(*temp)
    #
    # temp = list(zip(test[0], test[1], test[2]))
    # random.shuffle(temp)
    # test[0], test[1], test[2] = zip(*temp)

    train = [np.array(train[0]), np.array(train[1]), np.array(train[2])]
    val = [np.array(val[0]), np.array(val[1]), np.array(val[2])]
    test = [np.array((test[0])), np.array(test[1]), np.array(test[2])]
    feature_column = [sparse_feature(feat='user', feat_num=num_user + 1),
                      sparse_feature(feat='item', feat_num=num_item + 1)]

    print('数据预处理结束')
    return feature_column, train, val, test


# create_dataset('D:/data/ml-1m/ratings.dat', threshold=2, k=100)
