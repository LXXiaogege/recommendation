import numpy as np

"""
CART算法

回归树与分类树的区别：
样本输出：分类树离散值，回归树连续值
连续值处理方法： 分类树Gini系数，回归树误差平方最小化
预测方式： 分类树采用叶子结点概率最大的类别，回归树采用叶子结点的均值


基尼系数：表示集合的不确定性（纯度），系数越大不确定性却大，越小，纯度越高。
Gini(D):表示在集合D中取出两个样本其标记类别不一致的概率。
Gini(D,A):表示用特征A划分集合D之后集合D的不确定性。

"""


class DecisionNode:
    """
    决策节点
    """

    def __init__(self, feature_i=None, threshold=None,
                 value=None, true_branch=None, false_branch=None):
        """

        :param feature_i:  特征id（index）
        :param threshold: 特征的阈值
        :param value: 叶节点上的值（分类树为离散型的，回归树为连续性的）
        :param true_branch: # 左子树
        :param false_branch: # 右子树
        """
        self.feature_i = feature_i
        self.threshold = threshold
        self.value = value
        self.true_branch = true_branch
        self.false_branch = false_branch


class DecisionTree(object):
    def __init__(self, min_samples_split=2, min_impurity=1e-7,
                 max_depth=float("inf"), loss=None):
        """

        :param min_samples_split: 划分样本最小值
        :param min_impurity: 不确定性，杂质
        :param max_depth: 树的最大深度
        :param loss: 损失
        """
        self.root = None  # 决策树根节点
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self._impurity_calculation = None  # 切割树的方法，gini系数等
        self._leaf_value_calculation = None  # 树节点取值的方法，分类树：选取出现最多次数的值，回归树：取所有值的平均值
        self.one_dim = None  # If y is one-hot encoded (multi-dim) or not (one-dim)
        self.loss = loss  # If Gradient Boost

    def fit(self, X, y, loss=None):
        self.one_dim = len(np.shape(y)) == 1  # one_dim为bool型
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, current_depth=0):
        """
        生成决策树，CART递归生成二叉树的过程
        :return:
        """
        # Check if expansion of y is needed
        if len(np.shape(y)) == 1:
            y = np.expand_dims(y, axis=1)
        # Add y as last column of X
        Xy = np.concatenate((X, y), axis=1)
        n_samples, n_features = np.shape(X)

        if n_samples >= self.min_samples_split and current_depth <= self.max_depth:
            # 计算每个特征的impurity，选择impurity最小的特征
            for feature_i in range(n_features):
                # All values of feature_i ？？？？？？？？？？？？？
                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                unique_values = np.unique(feature_values)
