import numpy as np


class BaseLayer:
    def __init__(self):
        self.trainable = False
        self.weights= np.random.randn(10,10)
