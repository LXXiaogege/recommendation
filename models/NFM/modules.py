from tensorflow.keras.layers import Layer, Dense, BatchNormalization


class Deep(Layer):
    def __init__(self, hidden_units):
        super(Deep, self).__init__()
        self.dnn = [Dense(units, activation='relu') for units in hidden_units]
        self.bn = BatchNormalization()

    def call(self, inputs, *args, **kwargs):
        x = inputs
        for layer in self.dnn:
            x = layer(x)
            # x = self.bn(x)
        return x
