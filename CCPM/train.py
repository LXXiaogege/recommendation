import pandas as pd

from data_preprocess.avazu import create_avazu_dataset
from model import CCPM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping

path = '../data/avazu-ctr-prediction'

embed_dim = 11
conv_kernel_width = (6, 5, 3)
conv_filters = (4, 4, 4)

learning_rate = 0.001
batch_size = 64
epochs = 20

# samples_num 尽量大，避免过拟合
(train_x, train_y), (val_x, val_y), test_x, features_columns = create_avazu_dataset(path, read_part=True,
                                                                                    samples_num=1000000,
                                                                                    embed_dim=embed_dim)

model = CCPM(feat_column=features_columns, conv_kernel_width=conv_kernel_width, conv_filters=conv_filters)
model.compile(optimizer=Adam(), loss=binary_crossentropy)
model.fit(x=train_x, y=train_y, batch_size=batch_size, epochs=epochs, validation_split=0.1, shuffle=True,
          callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)])

result = model.predict(test_x, batch_size=10000)
pd.Series(result.flatten()).to_csv('result.csv', header=None, index=False)
