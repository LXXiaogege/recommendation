def sparse_faeture_dict(feat_name, feat_num, embed_dim=4):
    """
    为每个离散变量建立信息字典
    :param feat_name: 特征名
    :param feat_num: 特征数
    :param embed_dim: 特征维度
    :return:
    """
    return {"feat_name": feat_name, "feat_num": feat_num, "embed_dim": embed_dim}


def continue_feature_dict(feat_name):
    """
    为每个连续变量建立信息字典
    :param feat_name: 特证名
    :return:
    """
    return {"feat_name": feat_name}
