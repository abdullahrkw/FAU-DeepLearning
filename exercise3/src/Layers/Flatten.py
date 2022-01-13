import numpy as np


class Flatten:
    def __init__(self):
        self.trainable=False

    def forward(self, inpT):
        self.inpT = inpT
        batch_size = inpT.shape[0]
        return np.reshape(inpT, (batch_size, np.prod(inpT.shape[1:])))

    def backward(self, error_tensor):
        return np.reshape(error_tensor, self.inpT.shape)

