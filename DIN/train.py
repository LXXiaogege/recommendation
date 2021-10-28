import os
from data_preprocess.amazon_electronic_utils import create_amazon_electronic_dataset
from DIN.model import DIN
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import EarlyStopping

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

if __name__ == '__main__':
    # ========================= Hyper Parameters =======================
    file = r'D:\data\amazon_electronics\remap.pkl'
    maxlen = 20

    embed_dim = 8
    att_hidden_units = [80, 40]
    ffn_hidden_units = [256, 128, 64]
    dnn_dropout = 0.5
    att_activation = 'sigmoid'
    ffn_activation = 'prelu'

    learning_rate = 0.001
    batch_size = 4096
    epochs = 50

    feature_columns, behavior_list, (train_X, train_y), (val_X, val_y), (
        test_X, test_y) = create_amazon_electronic_dataset(file, embed_dim, maxlen)
    model = DIN(feature_columns, behavior_list, att_hidden_units, ffn_hidden_units, att_activation,
                ffn_activation, maxlen, dnn_dropout)

    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate),
                  metrics=[AUC()])

    model.fit(
        x=train_X, y=train_y, batch_size=batch_size, epochs=epochs,
        callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)],  # checkpoint
        validation_data=(val_X, val_y)
    )

    print("Model AUC in the test dataset ：", model.evaluate(x=test_X, y=test_y, batch_size=batch_size)[1])
