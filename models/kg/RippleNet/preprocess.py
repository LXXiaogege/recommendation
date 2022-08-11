import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tqdm import tqdm


def read_item_index_to_entity_id_file():
    """
    只取item_index2entity_id_rehashed.txt文件里的movie item作为数据集，因为其他item在知识图谱中不存在
    item_index2entity_id_rehashed.txt field: item_index , satori_id, 对rating.dat文件其中2445个电影的重新索引（可能只是图谱中只能找到这2445个电影的entity），第一列为movieId，
    第二列为知识图谱中该movie的id

    item_index_old2new:  key为第一列 movieId，value为 movie的重新索引
    entity_id2index： key为entity id ， value为movie的重新索引

    item的 new index与对应 在entity中的id是一样的，因此后面seed直接把item的index作为seed
    :return:
    """
    file = os.path.join(root_path, 'item_index2entity_id_rehashed.txt')
    print('reading item index to entity id file: ' + file + ' ...')
    i = 0
    for line in open(file, encoding='utf-8').readlines():
        item_index = line.strip().split('\t')[0]
        satori_id = line.strip().split('\t')[1]
        item_index_old2new[item_index] = i
        entity_id2index[satori_id] = i
        i += 1


def reindex_kg():
    """
    以entity_id2index（item movie）为基础，扩充知识图谱信息
    """
    print('reindex kg')
    kg1 = pd.read_csv(os.path.join(root_path, 'kg_part1_rehashed.txt'), sep='\t', names=['head', 'relation', 'tail'])
    kg2 = pd.read_csv(os.path.join(root_path, 'kg_part2_rehashed.txt'), sep='\t', names=['head', 'relation', 'tail'])
    kg = pd.concat([kg1, kg2])

    # entity 不只是 item（movie） 还有 actor等与item有联系的实体
    new_index = len(entity_id2index)
    for idx, (head, tail) in tqdm(enumerate(zip(kg['head'], kg['tail'])), total=len(kg['head'])):
        head, tail = str(head), str(tail)
        if head not in entity_id2index:
            entity_id2index[head] = new_index
            new_index += 1
        if tail not in entity_id2index:
            entity_id2index[tail] = new_index
            new_index += 1
        kg.iloc[idx, 0] = entity_id2index[head]
        kg.iloc[idx, 2] = entity_id2index[tail]
    le = LabelEncoder()
    kg['relation'] = le.fit_transform(kg['relation'])
    kg.to_csv(os.path.join(root_path, 'kg_final.csv'), sep='\t', index=False)
    print('kg reindex okay')


def reindex_for_ratings(rating_threshold=4, k=1):
    print("reindex user and item,construct dataset")
    ratings_path = os.path.join(root_path, 'ratings.dat')
    ratings = pd.read_csv(ratings_path, sep='::', engine='python', names=['user', 'item', 'rating', 'timestamp'])
    del ratings['timestamp']

    # 删除没有与知识图谱建立实体链接的item
    invalid_index = []
    for idx, item in enumerate(ratings['item']):
        if str(item) not in item_index_old2new:
            invalid_index.append(idx)
    ratings.drop(invalid_index, inplace=True)
    ratings.reset_index(drop=True, inplace=True)

    # user
    le = LabelEncoder()
    ratings['user'] = le.fit_transform(ratings['user'])
    # item
    items = []
    for item in ratings['item']:
        items.append(item_index_old2new[str(item)])
    ratings['item'] = items

    item_set = set(item_index_old2new.values())

    # construct positive sample
    pos_index = []
    for idx, rate in enumerate(ratings['rating']):
        if rate >= rating_threshold:
            pos_index.append(idx)
    pos_samples = ratings.loc[pos_index].reset_index(drop=True)
    del pos_samples['rating']
    pos_samples['label'] = 1

    # construct negative sample, 选择用户没评价过的电影为负样本集
    users_rating_item = {}  # key：用户 , value: 用户评价过的样本
    for user, data in ratings.groupby('user'):
        users_rating_item[user] = set(data['item'])
    neg_samples = pd.DataFrame(columns=['user', 'item'])
    for user, data in pos_samples.groupby('user'):
        pos_len = len(data['item'])
        if len(list(item_set - users_rating_item[user])) < pos_len:
            continue
        # item_set - user_neg_item
        user_neg_item = np.random.choice(list(item_set - users_rating_item[user]), size=pos_len, replace=False)
        tem = pd.DataFrame({'user': [user] * pos_len * k, 'item': user_neg_item})
        neg_samples = pd.concat([neg_samples, tem])
    neg_samples['label'] = 0

    data = pd.concat([pos_samples, neg_samples]).reset_index(drop=True)
    data.to_csv(os.path.join(root_path, 'ratings_final.csv'), sep='\t', index=False)
    print("ratings.dat is okay")


if __name__ == '__main__':
    root_path = 'data/movie'

    # movie 在知识图谱中的entity id 与  new index查找表
    entity_id2index = dict()
    # movie old index 与 new index 查找字典
    item_index_old2new = dict()

    read_item_index_to_entity_id_file()
    reindex_for_ratings()
    reindex_kg()
