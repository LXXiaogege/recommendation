from tensorflow.keras.layers import Layer, Dense,ReLU


class ResidualUnits(Layer):
    """
    残差单元,MLP
    """

    def __init__(self, hidden_unit, dim_stack):
        super(ResidualUnits, self).__init__()
        self.layer_0 = Dense(hidden_unit, activation='relu')
        self.layer_1 = Dense(dim_stack, activation=None)
        self.relu = ReLU()

    def call(self, inputs, *args, **kwargs):
        outputs = self.layer_0(inputs)
        outputs = self.layer_1(outputs)
        outputs = self.relu(outputs)
        return outputs
