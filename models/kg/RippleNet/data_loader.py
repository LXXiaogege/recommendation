from collections import defaultdict

import numpy as np
import pandas as pd
import os
from config import Config
import joblib


def create_dataset(root_path):
    dataset_path = os.path.join(root_path, 'dataset.pkl')
    memories_h_path = os.path.join(root_path, 'memories_h.pkl')
    memories_r_path = os.path.join(root_path, 'memories_r.pkl')
    memories_t_path = os.path.join(root_path, 'memories_t.pkl')

    dataset = pd.read_csv(os.path.join(root_path, 'ratings_final.csv'), sep='\t', names=['user', 'item', 'label'])

    # construct ripple set
    ripple_seed_dict = defaultdict(list)  # key: user(int), value: pos_samples(list)
    for user, data in dataset.groupby('user'):
        ripple_seed_dict[user].extend(data.loc[data['label'] == 1, 'item'].tolist())
    entity_len, relation_len, ripple_set_dict = create_ripple_set(root_path, ripple_seed_dict)

    # shuffle
    dataset.sample(frac=1).reset_index(drop=True)

    if os.path.exists(dataset_path):
        print("load data")
        dataset = joblib.load(dataset_path)
        memories_h = joblib.load(memories_h_path)
        memories_r = joblib.load(memories_r_path)
        memories_t = joblib.load(memories_t_path)
    else:
        memories_h, memories_r, memories_t = [], [], []  # 三层嵌套：1: user, 2: hop, 3: h/r/t
        for idx, data in dataset.iterrows():
            tem_h, tem_r, tem_t = [], [], []
            for hop in range(Config.n_hop):
                tem_h.append(ripple_set_dict[data['user']][hop][0])
                tem_r.append(ripple_set_dict[data['user']][hop][1])
                tem_t.append(ripple_set_dict[data['user']][hop][2])
            memories_h.append(tem_h)
            memories_r.append(tem_r)
            memories_t.append(tem_t)
        del dataset['user']
        joblib.dump(memories_h, memories_h_path)
        joblib.dump(memories_r, memories_r_path)
        joblib.dump(memories_t, memories_t_path)
        joblib.dump(dataset, dataset_path)

    print('spilt data')
    labels = dataset['label'].values
    items = dataset['item'].values

    data_len = len(dataset)
    test_len = int(data_len * 0.2)
    memories_h_train, memories_h_test = memories_h[:-test_len], memories_h[-test_len:]
    memories_r_train, memories_r_test = memories_r[:-test_len], memories_r[-test_len:]
    memories_t_train, memories_t_test = memories_t[:-test_len], memories_t[-test_len:]

    memories_train = [memories_h_train, memories_r_train, memories_t_train]
    memories_test = [memories_h_test, memories_r_test, memories_t_test]
    items_train, items_test = items[:-test_len], items[-test_len:]
    labels_train, labels_test = labels[:-test_len], labels[-test_len:]

    return (memories_train, memories_test, items_train, items_test, labels_train, labels_test), entity_len, relation_len


def create_ripple_set(root_path, ripple_seed_dict):
    print("create ripple set")
    kg = pd.read_csv(os.path.join(root_path, 'kg_final.csv'), sep='\t')
    entity_len = kg['head'].max() + 1 if kg['head'].max() > kg['tail'].max() else kg['tail'].max() + 1
    relation_len = kg['relation'].max() + 1

    ripple_set_path = os.path.join(root_path, 'ripple_set.pkl')

    if os.path.exists(ripple_set_path):
        ripple_set_dict = joblib.load(ripple_set_path)
    else:
        kg_dict = defaultdict(list)  # key:(int)head, value: (list) [(relation,tail)..]
        for index, row in kg.iterrows():
            kg_dict[row['head']].append((row['relation'], row['tail']))  # defaultdict不需要判空

        # 为每个用户构建一个ripple set
        ripple_set_dict = defaultdict(list)  # key: user, value: [ [([head_list],[relation_list],[tail_list])],...]
        for user in ripple_seed_dict:
            head_set = ripple_seed_dict[user]  # 每hop的head集合
            for hop in range(Config.n_hop):  # propagation
                memories_h = []
                memories_r = []
                memories_t = []
                if hop != 0:
                    head_set = ripple_set_dict[user][-1][2]
                for head in head_set:
                    for relation, tail in kg_dict[head]:
                        memories_h.append(head)
                        memories_r.append(relation)
                        memories_t.append(tail)
                if len(memories_h) == 0:  # 没有tail节点，则保留上次结果
                    ripple_set_dict[user].append(ripple_set_dict[user][-1])
                else:
                    replace = len(memories_h) < Config.n_memory
                    indices = np.random.choice(len(memories_h), size=Config.n_memory, replace=replace)
                    memories_h = [memories_h[i] for i in indices]
                    memories_r = [memories_r[i] for i in indices]
                    memories_t = [memories_t[i] for i in indices]
                    ripple_set_dict[user].append((memories_h, memories_r, memories_t))
        joblib.dump(ripple_set_dict, ripple_set_path)
    print("ripple set has created")
    return entity_len, relation_len, ripple_set_dict


if __name__ == '__main__':
    create_dataset('data/movie')
