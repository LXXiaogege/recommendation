import sys

sys.path.append('/home/admin607/data/lx/news-rec')
from RippleNet.config import Config
from RippleNet.model import RippleNet
from data_loader import create_dataset
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
import tensorflow as tf
import numpy as np

root_path = 'data/movie'
(memories_train, memories_test, items_train, items_test, labels_train,
 labels_test), entity_len, relation_len = create_dataset(root_path)
train_data_len = len(labels_train)
test_data_len = len(labels_test)

memories_h_train = memories_train[0]
memories_r_train = memories_train[1]
memories_t_train = memories_train[2]

memories_h_test = memories_test[0]
memories_r_test = memories_test[1]
memories_t_test = memories_test[2]

items_train, items_test, labels_train, labels_test = np.array(items_train, dtype='int32'), \
                                                     np.array(items_test, dtype='int32'), \
                                                     np.array(labels_train, dtype='int32'), \
                                                     np.array(labels_test, dtype='int32')

model = RippleNet(entity_len, relation_len)
optimizer = Adam(learning_rate=Config.lr)
metric = AUC()
batch_size = Config.batch_size
epochs = Config.epochs
loss_list = []

print("training")
for i in range(epochs):
    print('epoch:', i)
    batch_count = 0
    start, end = 0, batch_size
    while end < train_data_len:
        h = memories_h_train[start:end]
        r = memories_r_train[start:end]
        t = memories_t_train[start:end]
        item = items_train[start:end]
        label = labels_train[start:end]

        with tf.GradientTape() as tape:
            y_pred, loss = model((h, r, t, item, label), training=True)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        batch_count += 1
        start = end
        end += batch_size
        loss_list.append(loss.numpy())
        if batch_count % 10 == 0:
            print("batch num :", batch_count, ", train avg loss :", np.mean(loss_list))
            loss_list = []

    # evaluation
    start, end = 0, batch_size
    auc_list = []
    while end < test_data_len:
        h = memories_h_test[start:end]
        r = memories_r_test[start:end]
        t = memories_t_test[start:end]
        item = items_test[start:end]
        label = labels_test[start:end]
        y_pred, loss = model((h, r, t, item, label), training=False)
        start = end
        end += batch_size
        metric.update_state(label, y_pred)
        auc = metric.result().numpy()
        auc_list.append(auc)
    print("epoch:", i, ' test avg auc:', np.mean(auc_list))
