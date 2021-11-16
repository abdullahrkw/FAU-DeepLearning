import numpy as np
class BaseLayer:

    def __init__(self):
        print("Base_layer_trainable")
        self.trainable = False
        self.weights= np.random.randn(10,10)
