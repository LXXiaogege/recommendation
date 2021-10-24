import pickle
import random

import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences


def sparseFeature(feat, feat_num, embed_dim=4):
    """
    create dictionary for sparse feature
    :param feat: feature name
    :param feat_num: the total number of sparse features that do not repeat
    :param embed_dim: embedding dimension
    :return:
    """
    return {'feat': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}


def denseFeature(feat):
    """
    create dictionary for dense feature
    :param feat: dense feature name
    :return:
    """
    return {'feat': feat}


def create_amazon_electronic_dataset(file, embed_dim, maxlen):
    print("data preprocess start !!!")
    """
    1. 读取数据
    """
    with open(r'D:\data\amazon_electronics\remap.pkl', 'rb') as f:
        reviews_df = pickle.load(f)  # review评论信息，按reviewerID排序
        cate_list = pickle.load(f)  # 物品类别按item id 排序（item category）
        user_count, item_count, cate_count, example_count = pickle.load(f)
    reviews_df = reviews_df
    """
    2. 更改列名,reviewerID->user_id , asin->item_id , unixReviewTime->time
    """
    reviews_df.columns = ['user_id', 'item_id', 'time']

    """
    3. 正负样本1:1，因此生成对应的负样本，并且产生用户历史行为序列；
    """
    train_data, val_data, test_data = [], [], []
    for user_id, hist in tqdm(reviews_df.groupby('user_id')):
        """
        按user_id分组，遍历user_id,hist 为对应user_id的数据
        """
        pos_list = hist['item_id'].tolist()  # 正样本

        def gen_neg():
            """
            随机生成一个负样本，根据数据还有业务需求来设计正负样本生成比例和方式。
            :return:
            """
            neg = pos_list[0]
            while neg in pos_list:
                neg = random.randint(0, item_count - 1)
            return neg

        neg_list = [gen_neg() for i in range(len(pos_list))]  # 生成负样本 ，正负样本比例1:1

        """
        正样本label为1，负样本label为0
        生成训练集、测试集、验证集，每个用户所有浏览的物品（共n个）前n-2个为训练集（正样本），并生成相应的负样本，
        每个用户共有n-3个训练集（第1个无浏览历史），第n个作为测试集。第n-1个作为验证集。
        数据集结构：[hist_i(历史记录序列),[pos_list[i](要预测的下一个浏览记录)，cate_list[pos_list[i]]](要预测的下一个浏览记录的类别)，1/0(正负样本标签))
        """
        hist = []  # 该用户的history，user评价过的商品id
        for i in range(1, len(pos_list)):
            # hist 生成每一次的浏览记录，即之前的浏览历史。
            hist.append([pos_list[i - 1], cate_list[pos_list[i - 1]]])  # hist: [[pos item id,item cate ]] 二维数组
            hist_i = hist.copy()  # 浅拷贝
            if i == len(pos_list) - 1:
                test_data.append([hist_i, [pos_list[i], cate_list[pos_list[i]]], 1])
                test_data.append([hist_i, [neg_list[i], cate_list[neg_list[i]]], 0])
            elif i == len(pos_list) - 2:
                val_data.append([hist_i, [pos_list[i], cate_list[pos_list[i]]], 1])
                val_data.append([hist_i, [neg_list[i], cate_list[neg_list[i]]], 0])
            else:
                train_data.append([hist_i, [pos_list[i], cate_list[pos_list[i]]], 1])
                train_data.append([hist_i, [neg_list[i], cate_list[neg_list[i]]], 0])

    """
    得到feature_columns：无密集数据，稀疏数据为item_id和cate_id；
    """
    # feature columns: [dense_features,sparse_features]
    feature_columns = [[],
                       [sparseFeature('item_id', item_count, embed_dim),
                        ]]  # sparseFeature('cate_id', cate_count, embed_dim)

    """
    生成用户行为列表，方便后续序列Embedding的提取，在此处，即item_id, cate_id；
    """
    # behavior , behavior_list单独抽出来为了做target attention之类的用户行为序列建模
    behavior_list = ['item_id']  # , 'cate_id'

    # shuffle
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    """
    6. 得到新的训练集、验证集、测试集，格式为：'hist', 'target_item', 'label'；
    """
    # create dataframe,list convert to dataFrame
    train = pd.DataFrame(train_data, columns=['hist', 'target_item', 'label'])
    val = pd.DataFrame(val_data, columns=['hist', 'target_item', 'label'])
    test = pd.DataFrame(test_data, columns=['hist', 'target_item', 'label'])

    # if no dense or sparse features, can fill with 0
    """
    7. 由于序列的长度各不相同，因此需要使用tf.keras.preprocessing.sequence.pad_sequences方法进行填充；
    """
    print('==================Padding===================')
    train_X = [np.array([0.] * len(train)), np.array([0] * len(train)),
               pad_sequences(train['hist'], maxlen=maxlen),
               np.array(train['target_item'].tolist())]
    train_y = train['label'].values
    val_X = [np.array([0] * len(val)), np.array([0] * len(val)),
             pad_sequences(val['hist'], maxlen=maxlen),
             np.array(val['target_item'].tolist())]
    val_y = val['label'].values
    test_X = [np.array([0] * len(test)), np.array([0] * len(test)),
              pad_sequences(test['hist'], maxlen=maxlen),
              np.array(test['target_item'].tolist())]
    test_y = test['label'].values
    print('============Data Preprocess End=============')
    return feature_columns, behavior_list, (train_X, train_y), (val_X, val_y), (test_X, test_y)
