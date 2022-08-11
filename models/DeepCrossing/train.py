from data_preprocess.criteo import create_criteo_dataset
from models.DeepCrossing.model import DeepCrossing
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.metrics import AUC

if __name__ == '__main__':
    file = r"D:\data\kaggle_ad_ctr\train.txt"
    embed_dim = 8

    hidden_units = [256, 128, 64]
    batch_size = 64
    epochs = 10
    feature_column, (train_X, train_y), (test_X, test_y) = create_criteo_dataset(file=file, embed_dim=embed_dim)
    model = DeepCrossing(feature_column=feature_column, hidden_units=hidden_units, res_dropout=0.5)
    model.compile(optimizer=Adam(), loss=binary_crossentropy, metrics=AUC())
    model.fit(x=train_X, y=train_y, batch_size=batch_size, epochs=epochs, validation_split=0.1, shuffle=True)
    print("Model AUC in the test dataset ：", model.evaluate(x=test_X, y=test_y, batch_size=batch_size)[1])
