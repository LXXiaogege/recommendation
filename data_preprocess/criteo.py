"""
criteo dataset preprocess
Dataset download ： https://ailab.criteo.com/ressources/
"""
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder
from data_preprocess.utils import sparse_faeture_dict
from sklearn.model_selection import train_test_split


def create_criteo_dataset(file, embed_dim=8, read_part=True, sample_num=100000, test_size=0.2):
    """

    :param file: 训练集文件路径
    :param embed_dim:  特征维度
    :param read_part: Bool是否读取一部分数据
    :param sample_num:  如果read_part=True读取样本数
    :param test_size: float测试集占比
    :return:
    """
    names = ['label', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13', 'C1', 'C2',
             'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18',
             'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26']
    if read_part:
        # iterator:返回TextFileReader对象，迭代获取块 chunks,names列名
        tfr = pd.read_csv(file, sep='\t', header=None, iterator=True, names=names)
        data = tfr.get_chunk(sample_num)
    else:
        data = pd.read_csv(file, sep='\t', header=None, names=names)

    continue_features = ['I' + str(i) for i in range(1, 14)]  # 连续特征
    sparse_features = ['C' + str(i) for i in range(1, 27)]  # 离散特征
    features = continue_features + sparse_features

    print("填充特征空字段")
    data[continue_features] = data[continue_features].fillna(0)
    data[sparse_features] = data[sparse_features].fillna('-1')  # 要填成字符类型确保为离散数据，如果填成int，后面则无法对他编码

    # 连续特征离散化(分箱),n_bins：离散后的桶个数，encode:编码方式，strategy:分箱的策略
    est = KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='uniform')
    data[continue_features] = est.fit_transform(data[continue_features])

    # 离散数据编码 （数字编码），  离散数据无大小意义可用ont-hot编码，有大小意义选择数字编码LabelEncoder、OrdinalEncoder等
    for feat in data[sparse_features]:
        le = LabelEncoder()
        data[feat] = le.fit_transform(data[feat])

    # 因为连续数据也已经转化为了离散型，因此全使用sparse_faeture_dict建立信息字典
    feature_column = [sparse_faeture_dict(feat_name=feat, feat_num=int(data[feat].max()), embed_dim=embed_dim) for feat
                      in features]

    #  test这里其实是作为验证集
    train, test = train_test_split(data, test_size=test_size)

    train_X = train[features].values.astype('int32')
    train_y = train['label'].values.astype('int32')
    test_X = test[features].values.astype('int32')
    test_y = test['label'].values.astype('int32')

    return feature_column, (train_X, train_y), (test_X, test_y)
