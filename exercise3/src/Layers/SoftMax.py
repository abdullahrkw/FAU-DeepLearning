import numpy as np

from Layers.Base import BaseLayer


class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self.y  = None

    def forward(self, inpT):
        e = np.exp(inpT - np.amax(inpT, axis=inpT.ndim-1, keepdims=True))
        eSum = e.sum(axis=e.ndim-1, keepdims=True)
        self.y = e/eSum
        return self.y

    def backward(self, errT):
        dx = self.y * errT
        s = dx.sum(axis=1, keepdims=True)
        dx -= self.y * s
        return dx
    