import pickle
import pandas as pd
import os


def to_df(file_path):
    """
    把数据转为pandas DataFrame对象
    :param file_path:
    :return:
    """
    with open(file_path, mode='r') as fin:
        df = {}
        i = 0
        for line in fin:
            df[i] = eval(line)  # eval执行一个字符串表达式，并返回表达式的值。
            i += 1
        df = pd.DataFrame.from_dict(df, orient='index')
        return df


if not os.path.isfile(r'D:\data\amazon_electronics\reviews.pkl'):
    reviews_df = to_df(r"D:\data\amazon_electronics\Electronics_5.json")

    # pickle保存持久化
    with open(r'D:\data\amazon_electronics\reviews.pkl', 'wb') as f:
        pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL)

    meta_df = to_df(r"D:\data\amazon_electronics\meta_Electronics.json")
    meta_df = meta_df[meta_df['asin'].isin(reviews_df['asin'].unique())]
    meta_df = meta_df.reset_index(drop=True)  # 重置DataFrame的index

    with open(r'D:\data\amazon_electronics\meta.pkl', 'wb') as f:
        pickle.dump(meta_df, f, pickle.HIGHEST_PROTOCOL)
