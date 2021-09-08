"""
CART算法
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
        :param min_impurity:
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

    def _build_tree(self):
        """
        生成决策树
        :return:
        """
        pass
