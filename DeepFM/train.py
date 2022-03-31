from data_preprocess.criteo import create_criteo_dataset
from DeepFM.model import DeepFM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.metrics import AUC

embed_dim = 8
hidden_units = [256, 128, 64]
lr = 0.0001
batch_size = 256
epochs = 10
feature_column, (train_X, train_y), (test_X, test_y) = create_criteo_dataset(file="D:/data/kaggle_ad_ctr/train.txt",
                                                                             embed_dim=embed_dim, read_part=True)

model = DeepFM(features_column=feature_column, hidden_units=hidden_units)
model.compile(optimizer=Adam(learning_rate=lr), loss=binary_crossentropy, metrics=[AUC()])
model.fit(x=train_X, y=train_y, batch_size=batch_size, epochs=epochs, validation_data=(test_X, test_y), shuffle=True)

print("model evaluate:", model.evaluate(x=test_X, y=test_y, batch_size=batch_size))
