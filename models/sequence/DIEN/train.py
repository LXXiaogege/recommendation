from models.sequence import DIEN
from models.sequence import create_amazon_electronic_dataset
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.metrics import AUC
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from matplotlib import pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':
    file = r'D:\data\amazon_electronics\remap.pkl'
    maxlen = 20
    embed_dim = 8
    learning_rate = 0.001
    batch_size = 32
    epochs = 10
    alpha = 1.0

    feature_columns, behavior_list, (train_X, train_y), (val_X, val_y), (
        test_X, test_y) = create_amazon_electronic_dataset(file, embed_dim, maxlen)
    model = DIEN(feature_columns=feature_columns, behavior_columns=behavior_list)
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate), metrics=[AUC()])

    history = model.fit(x=train_X, y=train_y, batch_size=batch_size, epochs=epochs,
                        callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)],
                        validation_data=(val_X, val_y)
                        )

    print("Model AUC in the test dataset ：", model.evaluate(x=test_X, y=test_y, batch_size=batch_size)[1])

    # 可视化训练过程
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    plot_model(model, to_file='DIEN.png', show_shapes=True)
