import pickle
import random

from tqdm import tqdm


def create_amazon_electronic_dataset(file, embed_dim, maxlen):
    print("data preprocess start !!!")
    """
    1. 读取数据
    """
    with open(r'D:\data\amazon_electronics\remap.pkl', 'rb') as f:
        reviews_df = pickle.load(f)  # review评论信息，按reviewerID排序
        cate_list = pickle.load(f)  # 物品类别按item id 排序（item category）
        user_count, item_count, cate_count, example_count = pickle.load(f)
    reviews_df = reviews_df
    """
    2. 更改列名,reviewerID->user_id , asin->item_id , unixReviewTime->time
    """
    reviews_df.columns = ['user_id', 'item_id', 'time']

    """
    3. 正负样本1:1，因此生成对应的负样本，并且产生用户历史行为序列；
    """
    train_data, val_data, test_data = [], [], []
    for user_id, hist in tqdm(reviews_df.groupby('user_id')):
        """
        按user_id分组，遍历user_id,hist 为对应user_id的数据
        """
        pos_list = hist['item_id'].tolist()  # 正样本

        def gen_neg():
            """
            随机生成一个负样本
            :return:
            """
            neg = pos_list[0]
            while neg in pos_list:
                neg = random.randint(0, item_count - 1)
            return neg

        neg_list = [gen_neg() for i in range(len(pos_list))]  # 生成负样本 ，正负样本比例1:1

        hist = []
        for i in range(1, len(pos_list)):
            hist.append([pos_list[i - 1], cate_list[pos_list[i - 1]]])  # hist: [[pos item id,item cate ]] 二维数组
            hist_i = hist.copy()  # 浅拷贝
