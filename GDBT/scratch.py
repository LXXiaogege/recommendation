import progressbar
import numpy as np

"""
进度条设置
"""
bar_widgets = [
    'Training: ', progressbar.Percentage(), ' ', progressbar.Bar(marker="-", left="[", right="]"),
    ' ', progressbar.ETA()
]

"""
损失函数
"""


class Loss(object):
    def loss(self, y_true, y_pred):
        return NotImplementedError()

    def gradient(self, y, y_pred):
        raise NotImplementedError()

    def acc(self, y, y_pred):
        return 0


class SquareLoss(Loss):
    def __init__(self): pass

    def loss(self, y, y_pred):
        return 0.5 * np.power((y - y_pred), 2)

    def gradient(self, y, y_pred):
        return -(y - y_pred)


class SoftMaxLoss(Loss):
    def gradient(self, y, p):
        return y - p


class GBDT:
    """
    GBDT中的树全为回归树
    """

    def __init__(self, n_estimators, learning_rate, min_samples_split,
                 min_impurity, max_depth, regression):
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

        # 回归树
        self.trees = []
