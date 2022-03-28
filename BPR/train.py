from tensorflow.keras.callbacks import EarlyStopping

from BPR.utils import create_dataset
from BPR.model import BPR
from tensorflow.keras.optimizers import Adam
from BPR.evaluate import evaluate

file = 'D:/data/ml-1m/ratings.dat'
batch_size = 500
epochs = 10
lr = 0.0001
top_N = 10

feature_column, train, val, test = create_dataset('D:/data/ml-1m/ratings.dat', threshold=2, k=100)
model = BPR(feat_column=feature_column)
model.compile(optimizer=Adam(learning_rate=lr), )

model.fit(x=train, batch_size=batch_size, epochs=epochs, shuffle=True, validation_data=(val, None),
          callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)])
hr, ndcg = evaluate(model, test, top_N)
print("hit rate:", hr, ";  ndcg:", ndcg)
