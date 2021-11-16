import numpy as np

class SoftMax(object):
    def __init__(self):
        self.trainable = False
        self.y  = None

    def forward(self, inpT):
        pred = np.zeros(inpT.shape, dtype=np.float64)
        for ele in range(inpT.shape[0]):
            e = np.exp(inpT[ele] - np.max(inpT[ele]))
            eSum = e.sum()
            pred[ele] = e/eSum
        self.y = pred
        return pred

    def backward(self, errT):
        dx = self.y * errT
        s = dx.sum(axis=1, keepdims=True)
        dx -= self.y * s
        return dx
    