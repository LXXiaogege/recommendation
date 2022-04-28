import json
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
from random import sample, choice

from tqdm import tqdm
from transformers import BertTokenizer

"""
behavior.tsv
Impression ID. The ID of an impression.
User ID. The anonymous ID of a user.
Time. The impression time with format "MM/DD/YYYY HH:MM:SS AM/PM". 2019年10月12日至11月22日
History. 用户的点击历史，按时间排序
Impressions. 该impression中的新闻列表，顺序被打乱，1点击，0未点击
"""


def parse_behavior(path, k=4):
    """
    k : 正负样本比，负样本：正样本
    behaviors_parsed.csv filed: user_id,history,impressions,labels
    """
    behavior_path = os.path.join(path, 'behaviors.tsv')
    behaviors_parsed_path = os.path.join(path, 'behaviors_parsed.csv')
    if os.path.exists(behaviors_parsed_path) is False:
        behaviors = pd.read_csv(behavior_path, sep='\t',
                                names=['impression_id', 'user_id', 'time', 'history', 'impressions'])
        # 暂时不考虑time
        del behaviors['impression_id'], behaviors['time']
        # user id
        user_id2index_path = os.path.join(path, 'user_id2index.pkl')
        if os.path.exists(user_id2index_path):
            le = joblib.load(user_id2index_path)
            behaviors['user_id'] = le.transform(behaviors['user_id'])
        else:
            le = LabelEncoder()
            behaviors['user_id'] = le.fit_transform(behaviors['user_id'])
            joblib.dump(le, user_id2index_path)

        # impressions:根据正负样本比构造候选序列

        behaviors = behaviors.astype('object')
        del_li = []
        for i in range(len(behaviors['impressions'])):
            impressions = behaviors['impressions'][i].split()
            negatives = [x for x in impressions if x.endswith('0')]
            if len(negatives) < k:  # 过滤负样本不够的数据
                del_li.append(i)
                continue
            positives = [x for x in impressions if x.endswith('1')]
            cond_seq = [choice(positives)]
            cond_seq.extend(sample(negatives, k))
            behaviors.at[i, 'impressions'] = [cond.split('-')[0] for cond in cond_seq]
        behaviors.drop(del_li, axis=0, inplace=True)
        behaviors.reset_index(drop=True, inplace=True)
        behaviors.insert(behaviors.shape[1], 'labels', str([1, 0, 0, 0, 0]))
        behaviors.to_csv(behaviors_parsed_path, index=False, sep='\t')


def parse_news(path, title_max_len=20, abstract_max_len=50, confidence_threshold=0.5):
    news_path = os.path.join(path, 'news.tsv')
    news_parsed_path = os.path.join(path, 'news_parsed.csv')
    if os.path.exists(news_parsed_path) is False:
        news = pd.read_csv(news_path, sep='\t', names=['id', 'category', 'subcategory', 'title', 'abstract', 'url',
                                                       'title_entities', 'abstract_entities'])
        del news['url']

        # category, subcategory
        sparse_feat = ['category', 'subcategory']
        le = LabelEncoder()
        for feat in sparse_feat:
            news[feat] = le.fit_transform(news[feat])

        # title,abstract
        news.fillna({'title': ' ', 'abstract': ' ', 'title_entities': '[]', 'abstract_entities': '[]'}, inplace=True)
        news = news.astype('object')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        for idx, (title, abstract, ti_entity, ab_entity) in tqdm(enumerate(zip(news['title'], news['abstract'],
                                                                               news['title_entities'],
                                                                               news['abstract_entities'])),
                                                                 total=len(news), desc='tokenize'):
            news.at[idx, 'title'] = tokenizer.encode(title, padding='max_length', max_length=title_max_len)
            news.at[idx, 'abstract'] = tokenizer.encode(abstract, padding='max_length', max_length=abstract_max_len)

            # 先按offset排序，去除一部分confidence低的，保留WikidataId
            ti_entity = sorted(json.loads(ti_entity), key=lambda i: i['OccurrenceOffsets'])
            news.at[idx, 'title_entities'] = [e['WikidataId'] for e in ti_entity if
                                              e['Confidence'] >= confidence_threshold]
            ab_entity = sorted(json.loads(ab_entity), key=lambda i: i['OccurrenceOffsets'])
            news.at[idx, 'abstract_entities'] = [e['WikidataId'] for e in ab_entity if
                                                 e['Confidence'] >= confidence_threshold]
        news.to_csv(news_parsed_path, index=False, sep='\t')


if __name__ == '__main__':
    root_path = r'D:\data\MINDSmall'
    print('behavior parse')
    parse_behavior(root_path)
    print('behavior parse success')
    print('news parse')
    parse_news(root_path)
    print('news parse success')
