import os

from keras.callbacks import EarlyStopping
from keras.metrics import AUC

from data_preprocess.criteo import create_criteo_dataset
from WD.model import WideDeep
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy

# 设置一个环境变量
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':
    # 设置GPU series number，如果有GPU
    # os.environ['CUDA_VISIBLE_DEVICES'] = '4'

    # 数据预处理参数
    file = r"D:\data\kaggle_ad_ctr\train.txt"
    read_part = True
    sample_num = 5000
    test_size = 0.2

    # 模型超参数
    embed_dim = 8
    dnn_dropout = 0.5
    hidden_units_num = [256, 128, 64]
    learning_rate = 0.001
    batch_size = 4096
    epochs = 100

    feature_columns, (X_train, y_train), (X_test, y_test) = create_criteo_dataset(file=file, embed_dim=embed_dim,
                                                                                  read_part=read_part,
                                                                                  sample_num=sample_num,
                                                                                  test_size=test_size)

    # GPU分布式训练 参考源代码。先不实现

    model = WideDeep(feature_columns, hidden_units_num=hidden_units_num, dnn_dropout=dnn_dropout)
    # model.summary()

    # compile():为训练做配置 ,metrics: 模型评价指标
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=binary_crossentropy, metrics=[AUC()])

    # callbacks()回调函数，在训练过程中某些点进行调用
    # EarlyStopping在没有改善时停止训练，实际训练的epochs <= 超参数设置的epochs
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
              callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)], validation_split=0.1
              )

    print('test AUC: %f' % model.evaluate(X_test, y_test, batch_size=batch_size)[1])
    model.summary()
