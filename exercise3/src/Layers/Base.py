import numpy as np


class BaseLayer(object):
    def __init__(self):
        self.trainable = False
        self.weights = np.random.randn(10,10)
        self.testing_phase = False

    def forward(self, inpT:np.ndarray):
        raise NotImplementedError

    def backward(self, errT:np.ndarray):
        raise NotImplementedError
