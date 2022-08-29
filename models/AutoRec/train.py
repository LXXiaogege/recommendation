from data_preprocess.movie_len import parse_ratings
from models.AutoRec.model import AutoRec

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.metrics import RootMeanSquaredError

root_path = r"../../data/ml-1m"
batch_size = 64
epochs = 10
lr = 0.001
num_units = 500
print("parse data")
dataset, num_out = parse_ratings(root_path)
print("parsed data success")
model = AutoRec(num_units=num_units, num_out=num_out)
model.compile(optimizer=Adam(learning_rate=lr), loss=mean_squared_error, metrics=[RootMeanSquaredError()])
model.fit(x=dataset, y=dataset, batch_size=batch_size, epochs=epochs, shuffle=False)
