import numpy as np


def evaluate(model, test, N):
    """
    evaluate model
    :param model:
    :param test: test dataset
    :param N: top N
    :return: hit rate , nDCG
    """
    pred_y = - model.predict(test)
    rank = pred_y.argsort().argsort()[:, 0]
    hr, ndcg = 0.0, 0.0
    for r in rank:
        if r < N:
            hr += 1
            ndcg += 1 / np.log2(r + 2)
    return hr / len(rank), ndcg / len(rank)
