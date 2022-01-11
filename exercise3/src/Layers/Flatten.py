import numpy as np


class Flatten:
    def __init__(self):
        self.trainable=False

    def forward(self, input_tensor):
        batch_size, width, height, depth = np.shape(input_tensor)
        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.depth = depth
        return np.reshape(input_tensor, (batch_size, width * height * depth))

    def backward(self, error_tensor):
        return np.reshape(error_tensor, (self.batch_size, self.width, self.height, self.depth))

