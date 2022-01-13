import numpy as np
from Layers.Base import BaseLayer


class Sigmoid(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, inpT: np.ndarray):
        self.activation = 1/(1 + np.exp(-inpT))
        return self.activation
  
    def backward(self, errT: np.ndarray):
        derv = self.activation*(1 - self.activation)
        return derv*errT
