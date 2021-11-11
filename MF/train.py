from MF.utils import create_ml_1m_dataset
from MF.model import MF
from tensorflow.keras.optimizers import Adam

if __name__ == '__main__':
    file = r'D:\data\ml-latest-small\ratings.csv'
    latent_dim = 32
    test_size = 0.2
    batch_size = 64
    epochs = 10
    use_bias = True
    features_column, (X_train, y_train), (X_test, y_test) = create_ml_1m_dataset(file, latent_dim, test_size)
    model = MF(features_column, use_bias=use_bias)
    model.compile(optimizer=Adam(), loss='mse', metrics=['mse'])

    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, validation_split=0.1)

    print("Metric MSE(均方误差) score: ", model.evaluate(X_test, y_test, batch_size=batch_size)[1])
