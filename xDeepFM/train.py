from data_preprocess.criteo import create_criteo_dataset
from model import xDeepFM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.metrics import AUC

file = "D:/data/kaggle_ad_ctr/train.txt"
embed_dim = 8
hidden_units = [256, 128, 64]
cin_hidden_units = [128, 128]
lr = 0.0001
batch_size = 256
epochs = 10
feature_column, (train_X, train_y), (test_X, test_y) = create_criteo_dataset(file=file,
                                                                             embed_dim=embed_dim, read_part=True)

model = xDeepFM(feature_column, hidden_units, cin_hidden_units)
model.compile(optimizer=Adam(lr), loss=binary_crossentropy, metrics=AUC())
model.fit(x=train_X, y=train_y, batch_size=batch_size, epochs=epochs, validation_data=(test_X, test_y), shuffle=True)
print("model evaluate:", model.evaluate(x=test_X, y=test_y))
