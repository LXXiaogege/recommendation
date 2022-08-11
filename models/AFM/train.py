from data_preprocess.criteo import create_criteo_dataset
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.metrics import AUC
from model import AFM

file = "D:/data/kaggle_ad_ctr/train.txt"
embed_dim = 8
attention_dense_dim = 8
hidden_units = [256, 128, 64]
lr = 0.0001
batch_size = 256
epochs = 10
#  mode ： 'afm' or 'fm'
mode = 'fm'

feature_column, (train_X, train_y), (test_X, test_y) = create_criteo_dataset(file=file, embed_dim=embed_dim,
                                                                             read_part=True)
model = AFM(feature_column, attention_dense_dim, mode)
model.compile(optimizer=Adam(lr), loss=mean_squared_error, metrics=AUC())
model.fit(x=train_X, y=train_y, batch_size=batch_size, epochs=epochs, validation_data=(test_X, test_y), shuffle=True)
print("model evaluate:", model.evaluate(x=test_X, y=test_y))
