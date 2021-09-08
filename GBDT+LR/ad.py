import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.ensemble import GradientBoostingClassifier


def data_proprecess():
    """
    因为测试集没有Label所以可以忽略。。在这里没什么用，原本是提交结果到Kaggle的

    Data Filed：
              ID， 广告id
              Label,是否被点击
              L1~L13,数值型特征
              C1~C26, Category特征，匿名化，做了Hash映射
    :return: 读取原始数据删除ID Column， 填充空字段
    """
    path = 'data/'
    print('读取数据')
    if not os.path.exists(path + 'data.csv'):
        df_train = pd.read_csv(path + 'train.csv')
        # df_test = pd.read_csv(path + 'test.csv')
        print('读取结束')
        # 删除Id列
        data = df_train.drop(['Id'], axis=1, inplace=True)
        # df_test.drop(['Id'], axis=1, inplace=True)

        # df_test['Label'] = -1  # 这个标签值只用来区分训练集与测试集

        # data = pd.concat([df_train, df_test])  # 把训练集 测试集拼到一块
        data = data.fillna(-1)  # 把空数据Na置为 -1
        data.to_csv('data/data.csv')
        return data
    else:
        data = pd.read_csv(path + 'data.csv')
        print("读取结束")
        return data


def lr(data, continuous_feature, category_feature):
    """

    :param data:
    :param continuous_feature: 整数型特征
    :param category_feature:  Category离散特征
    :return:
    """
    print("（整数值型特征 ）continuous_feature 归一化")
    scaler = MinMaxScaler()  # 默认归一化多0-1，可修改
    for col in continuous_feature:
        data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1))
    print("continuous_feature 归一化结束 ")

    print("category_feature 离散特征ont-hot编码")
    for col in category_feature:
        onehot_feats = pd.get_dummies(data[col], prefix=col)  # shape: 1999*79,prefix列名前缀
        data.drop([col], axis=1, inplace=True)  # inplace 返回一个副本，否则需要新建一个对象存储
        data = pd.concat([data, onehot_feats], axis=1)
    print("one-hot编码结束")

    train_data = data[data['Label'] != -1]  # 训练集
    train_data_label = train_data.pop('Label')  # 训练集标签
    # test_data = data[data['Label'] == 1]
    # test_data.drop('Label', axis=1, inplace=True)  # 删除Label Column

    # 划分训练集与验证集
    X_train, X_val, y_train, y_val = train_test_split(train_data, train_data_label, test_size=0.1, random_state=2018)
    clf = LogisticRegression(solver='sag', max_iter=100)
    print("开始训练")
    clf.fit(X_train, y_train)
    print("训练结束")

    print("计算交叉熵")
    train_loss = log_loss(y_true=train_data_label, y_pred=clf.predict_proba(train_data), labels=[0, 1])
    val_loss = log_loss(y_true=y_val, y_pred=clf.predict_proba(X_val), labels=[0, 1])
    print("训练集平均交叉熵损失：", train_loss, "验证集平均交叉熵损失:", val_loss)

    print("验证集准确率", clf.score(X_val, y_val))


def gbdt():
    pass


if __name__ == '__main__':
    data = data_proprecess()
    continuous_feature = ['I'] * 13
    continuous_feature = [col + str(i + 1) for i, col in enumerate(continuous_feature)]

    category_feature = ['C'] * 26
    category_feature = [col + str(i + 1) for i, col in enumerate(category_feature)]

    lr(data, continuous_feature, category_feature)
    gbdt()
