import os
from data_preprocess.amazon_electronic_utils import create_amazon_electronic_dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

if __name__ == '__main__':
    # ========================= Hyper Parameters =======================
    file = r'D:\data\amazon_electronics\remap.pkl'
    maxlen = 20

    embed_dim = 8
    att_hidden_units = [80, 40]
    ffn_hidden_units = [256, 128, 64]
    dnn_dropout = 0.5
    att_activation = 'sigmoid'
    ffn_activation = 'prelu'

    learning_rate = 0.001
    batch_size = 4096
    epochs = 5

    feature_columns, behavior_list, (train_X, train_y), (val_X, val_y), (
    test_X, test_y) = create_amazon_electronic_dataset(file, embed_dim, maxlen)
