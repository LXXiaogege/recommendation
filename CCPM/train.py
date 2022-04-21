from data_preprocess.avazu import create_avazu_dataset
from model import CCPM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy

path = 'D:/data/avazu-ctr-prediction'

embed_dim = 11
conv_kernel_width = (6, 5, 3)
conv_filters = (4, 4, 4)

learning_rate = 0.001
batch_size = 32
epochs = 10

# samples_num 尽量大，避免过拟合
(train_x, train_y), (val_x, val_y), features_columns = create_avazu_dataset(path, read_part=True, samples_num=100000,
                                                                            embed_dim=embed_dim)

model = CCPM(feat_column=features_columns, conv_kernel_width=conv_kernel_width, conv_filters=conv_filters)
model.compile(optimizer=Adam(), loss=binary_crossentropy)
model.fit(x=train_x, y=train_y, batch_size=batch_size, epochs=epochs, validation_split=0.1, shuffle=True)

# predict failed ：avazu 数据集太大无法完全加载，导致test data 里的某些特征超过了用于训练的数据集中的embedding vocabulary
# model.predict()
