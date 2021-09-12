"""
network component
"""
from tensorflow.keras.layers import Layer


class Wide(Layer):
    """
    Wide Component
    """
    def __init__(self):
        super(Wide, self).__init__()
