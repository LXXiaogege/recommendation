import progressbar
import numpy as np
from models.GDBT.decision_tree_model import RegressionTree
from sklearn import datasets
from sklearn.model_selection import train_test_split

"""
参考文章 : https://zhuanlan.zhihu.com/p/32181306
"""

"""
进度条设置
"""
bar_widgets = [
    'Training: ', progressbar.Percentage(), ' ', progressbar.Bar(marker="-", left="[", right="]"),
    ' ', progressbar.ETA()
]


class Loss(object):
    """
    损失函数
    """

    def loss(self, y_true, y_pred):
        return NotImplementedError()

    def gradient(self, y, y_pred):
        raise NotImplementedError()

    def acc(self, y, y_pred):
        return 0


class SquareLoss(Loss):
    """
    平方损失函数
    """

    def __init__(self): pass

    def loss(self, y, y_pred):
        return 0.5 * np.power((y - y_pred), 2)

    def gradient(self, y, y_pred):
        return -(y - y_pred)


class SoftMaxLoss(Loss):
    """
    softMax损失函数
    """

    def gradient(self, y, p):
        return y - p


class GBDT:
    """
    GBDT中的树全为回归树，
    创建n_estimators棵树的GBDT，注意这里的分类问题也使用回归树，利用残差去学习概率
    """

    def __init__(self, n_estimators, learning_rate, min_samples_split,
                 min_impurity, max_depth, regression):
        """

        :param n_estimators: 树的个数
        :param learning_rate: 学习率
        :param min_samples_split:  回归树参数
        :param min_impurity: 回归树参数
        :param max_depth: 回归树参数
        :param regression: bool 是回归还是分类
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self.regression = regression

        # 进度条
        self.bar = progressbar.ProgressBar(widgets=bar_widgets)

        if self.regression:
            self.loss = SquareLoss()  # 回归损失函数
        else:
            self.loss = SoftMaxLoss()  # 分类损失函数

        # 回归树，分类问题也使用回归树，利用残差去学习概率
        self.trees = []
        for i in range(self.n_estimators):
            self.trees.append(RegressionTree(min_samples_split=self.min_samples_split,
                                             min_impurity=self.min_impurity,
                                             max_depth=self.max_depth
                                             ))

    def fit(self, X, y):
        """
        利用残差学习，不断让下一棵树拟合上一颗树的"残差"(梯度)，而"残差"是由梯度求出
        :param X: 训练集data
        :param y: 训练集Label
        :return:  model
        """
        self.trees[0].fit(X, y)
        y_pred = self.trees[0].predict(X)
        for i in self.bar(range(1, self.n_estimators)):
            gradient = self.loss.gradient(y, y_pred)
            self.trees[i].fit(X, gradient)
            y_pred = np.multiply(self.learning_rate, self.trees[i].predict(X))  # np.multiply乘法运算

    def predict(self, X):
        """
        for循环的过程就是汇总各棵树的残差得到最后的结果
        :param X: 测试集X
        :return: 预测结果y
        """
        y_pred = self.trees[0].predict(X)

        if self.regression:
            # 回归树
            for i in self.bar(range(1, self.n_estimators)):
                y_pred = np.multiply(self.learning_rate, self.trees[i].predict(X))
        else:
            # 分类树
            y_pred = np.exp(y_pred) / np.expand_dims(np.sum(np.exp(y_pred), axis=1), axis=1)
            y_pred = np.argmax(y_pred, axis=1)
        return y_pred


class GBDTRegressor(GBDT):
    def __init__(self, n_estimators=200, learning_rate=0.5, min_samples_split=2,
                 min_var_red=1e-7, max_depth=4, debug=False):
        super(GBDTRegressor, self).__init__(n_estimators=n_estimators,
                                            learning_rate=learning_rate,
                                            min_samples_split=min_samples_split,
                                            min_impurity=min_var_red,
                                            max_depth=max_depth,
                                            regression=True)


def to_categorical(x, n_col=None):
    """ One-hot encoding of nominal values """
    if not n_col:
        n_col = np.amax(x) + 1
    one_hot = np.zeros((x.shape[0], n_col))
    one_hot[np.arange(x.shape[0]), x] = 1
    return one_hot


class GBDTClassifier(GBDT):
    def __init__(self, n_estimators=200, learning_rate=.5, min_samples_split=2,
                 min_info_gain=1e-7, max_depth=2, debug=False):
        super(GBDTClassifier, self).__init__(n_estimators=n_estimators,
                                             learning_rate=learning_rate,
                                             min_samples_split=min_samples_split,
                                             min_impurity=min_info_gain,
                                             max_depth=max_depth,
                                             regression=False)

    def fit(self, X, y):
        y = to_categorical(y)
        super(GBDTClassifier, self).fit(X, y)


def accuracy_score(y_true, y_pred):
    """ Compare y_true to y_pred and return the accuracy """
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy


if __name__ == '__main__':
    print("GBDT Classification")
    data = datasets.load_iris()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    clf = GBDTClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("准确率：", acc)
