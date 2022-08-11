from data_preprocess.criteo import create_criteo_dataset
from models.FM import FM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.metrics import AUC

if __name__ == '__main__':
    file = r"D:\data\kaggle_ad_ctr\train.txt"
    latent_dim = 8
    batch_size = 64
    epochs = 10
    feature_columns, (X_train, y_train), (X_test, y_test) = create_criteo_dataset(file=file, embed_dim=latent_dim)
    model = FM(feature_columns=feature_columns, latent_dim=latent_dim)
    model.compile(optimizer=Adam(), loss=binary_crossentropy, metrics=AUC())
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    print('test AUC: %f' % model.evaluate(X_test, y_test, batch_size=batch_size)[1])
