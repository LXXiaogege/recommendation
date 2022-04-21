import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from data_preprocess.utils import sparse_faeture_dict


def create_avazu_dataset(path, read_part=True, samples_num=5000, embed_dim=8):
    train_path = path + '/train.csv'
    print('加载数据集')
    if read_part:
        train_data = pd.read_csv(train_path, nrows=samples_num)
    else:
        train_data = pd.read_csv(path)

    # 暂时舍弃第C20， 数据上下差别太大应该预处理
    train_data[['C20', 'C21']] = train_data[['C21', 'C20']]
    sparse_features = train_data.columns.tolist()[3:-1]
    need_transform_feat = train_data.columns.tolist()[5:-10]

    # 也可以 Hash Encoder
    le = LabelEncoder()
    for feat in need_transform_feat:
        train_data[feat] = le.fit_transform(train_data[feat])

    features_columns = [sparse_faeture_dict(feat_name=feat, feat_num=train_data[feat].max() + 1, embed_dim=embed_dim)
                        for feat in sparse_features]

    train, val = train_test_split(train_data, test_size=0.2, shuffle=True)
    train_x = train[sparse_features].values.astype('int32')
    train_y = train['click'].values.astype('int32')
    val_x = val[sparse_features].values.astype('int32')
    val_y = val['click'].values.astype('int32')

    print('加载数据完成')
    return (train_x, train_y), (val_x, val_y), features_columns


def avazu_dataset_analysis(path, read_part=True, samples_num=5000):
    """
    https://blog.csdn.net/u014128608/article/details/93393175
    """
    train_data = pd.read_csv(path, nrows=samples_num)
    # dataset summary
    # print(train_data.info(verbose=True))

    # 查看缺失值
    # print(train_data.isnull().any())

    # 数据分布
    print(train_data.describe())
    for i in range(14, 22):
        sns.countplot(train_data['C' + str(i)])  # 更换filed，查看不同filed的分布
        plt.show()


# if __name__ == '__main__':
#     root_path = 'D:/data/avazu-ctr-prediction/train.csv'

    # dataset analysis
    # avazu_dataset_analysis(root_path)

    # create_avazu_dataset(root_path)
