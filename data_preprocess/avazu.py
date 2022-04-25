import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from data_preprocess.utils import sparse_faeture_dict

# todo 有的好的硬件再测试吧，test代码在kaggle上
"""
11天的数据 : train.csv 4kw+数据， test.csv： 400w+数据
avazu dataset field:
id : 广告id , id 算是离散特征
click： 0/1
hour: format is YYMMDDHH, so 14091123 means 23:00 on Sept. 11, 2014 UTC.
banner_pos
site_id
site_domain
site_category
app_id
app_domain
app_category
device_id
device_ip
device_model
device_type
device_conn_type
C14-C21 -- anonymized categorical variables
"""


def create_avazu_dataset(path, read_part=True, samples_num=5000, embed_dim=8):
    print('数据预处理开始')
    sparse_features = ['hour', 'id', 'C1', 'banner_pos', 'site_id', 'site_domain',
                       'site_category', 'app_id', 'app_domain', 'app_category', 'device_id',
                       'device_ip', 'device_model', 'device_type', 'device_conn_type', 'C14',
                       'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']

    train_path = path + '/train.gz'
    test_path = path + '/test.gz'
    print('加载数据集')
    if read_part:
        train_data = pd.read_csv(train_path, nrows=samples_num)
    else:
        train_data = pd.read_csv(train_path)
    test_x = pd.read_csv(test_path)

    # hour, 只有14年10月11天的数据，year,month没必要做特征
    train_data['hour'] = train_data['hour'].apply(str)
    train_data['hour'] = train_data['hour'].map(lambda x: int(x[6:8]))  # int强转去掉字符串前的0
    test_x['hour'] = test_x['hour'].apply(str)
    test_x['hour'] = test_x['hour'].map(lambda x: int(x[6:8]))
    print('加载数据完成')
    print('Sparse feature encode')
    # sparse features
    le = LabelEncoder()
    for feat in sparse_features:
        all_class = pd.concat([train_data[feat], test_x[feat]]).unique()
        le.fit(all_class)
        train_data[feat] = le.transform(train_data[feat])
        test_x[feat] = le.transform(test_x[feat])

    print('Sparse feature encode succeed')
    # save LabelEncoder model for test
    # joblib.dump(le, 'label_encoder.model')
    # sparse_faeture_dict(feat_name='day', feat_num=32, embed_dim=embed_dim)
    # sparse_faeture_dict(feat_name='hour', feat_num=24, embed_dim=embed_dim)
    features_columns = [sparse_faeture_dict(feat_name=feat, feat_num=train_data[feat].max() + 1, embed_dim=embed_dim)
                        for feat in sparse_features]

    train, val = train_test_split(train_data, test_size=0.2, shuffle=True)
    train_x = train[sparse_features].values.astype('int32')
    train_y = train['click'].values.astype('int32')
    val_x = val[sparse_features].values.astype('int32')
    val_y = val['click'].values.astype('int32')
    test_x = test_x[sparse_features].values.astype('int32')

    print('数据预处理完成')
    return (train_x, train_y), (val_x, val_y), test_x, features_columns


def avazu_dataset_analysis(path, read_part=True, samples_num=5000):
    """
    https://blog.csdn.net/u014128608/article/details/93393175
    """
    train_path = path + '/train.csv'
    train_data = pd.read_csv(train_path, nrows=samples_num)
    # dataset summary
    # print(train_data.info(verbose=True))
    # 查看缺失值
    # print(train_data.isnull().any())
    # 数据分布
    # print(train_data.describe())
    # for i in range(14, 22):
    #     sns.countplot(train_data['C' + str(i)])  # 更换filed，查看不同filed的分布
    #     plt.show()
    # print(train_data['C1'].unique())


def gen_commit_result(path):
    res = pd.read_csv(path + '/res.csv')
    result = pd.read_csv(path + '/result.csv')
    res['click'] = result['click']
    res.to_csv('res.csv', index=False)


# if __name__ == '__main__':
#     root_path = 'D:/data/avazu-ctr-prediction'
    # dataset analysis
    # avazu_dataset_analysis(root_path)

    # create_avazu_dataset(root_path)

    # gen_commit_result(root_path)
