import pandas as pd
from tqdm import tqdm


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


def create_ml_1m_dataset(file, latent_dim=4, test_size=0.2):
    """
    userId,movieId,rating,timestamp
    增加一个用户的平均rating（avg_score）来做特征
    """
    data_df = pd.read_csv(file, sep=',')
    data_df['avg_score'] = data_df.groupby('userId')['rating'].transform('mean')  # 算每个user的平均rating

    # feature column
    user_num, item_num = data_df['userId'].max(), data_df['movieId'].max()
    features_column = [denseFeature(feat='avg_score'),
                       [sparseFeature(feat='userId', feat_num=user_num, embed_dim=latent_dim),
                        sparseFeature(feat='movieId', feat_num=item_num, embed_dim=latent_dim)]]

    # spilt train,test dataset
    watch_count = data_df.groupby('userId')['movieId'].agg('count')  # 统计每个用户看过多少个电影

    # 划分测试集，0.2 ，每个用户的后20%数据作为测试集
    test_df = pd.concat([
        data_df[data_df.userId == i].iloc[int(0.8 * watch_count[i]):] for i in tqdm(watch_count.index)], axis=0)
    test_df = test_df.reset_index()
    train_df = data_df.drop(labels=test_df['index'])  # 去除测试集
    train_df = train_df.drop(['timestamp'], axis=1).sample(frac=1.).reset_index(drop=True)
    test_df = test_df.drop(['index', 'timestamp'], axis=1).sample(frac=1.).reset_index(
        drop=True)  # sample 随机抽取数据，起shuffle的作用

    X_train = [train_df['avg_score'].values, train_df['userId'].values, train_df['movieId'].values]
    X_test = [test_df['avg_score'].values, test_df['userId'].values, test_df['movieId'].values]
    y_train = train_df['rating'].values.astype('int64')
    y_test = test_df['rating'].values.astype('int64')

    return features_column, (X_train, y_train), (X_test, y_test)
