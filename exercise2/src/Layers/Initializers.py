import numpy as np

class Constant:
    def __init__(self, value=0.1):
        self.value = value

    # Because it is usually for Bias.
    def initialize(self, weights_shape, fan_in, fan_out):
        return np.ones((fan_out)) * self.value

class UniformRandom:
    def initialize(self, weights_shape,fan_in, fan_out):
        return np.random.uniform(size=weights_shape)

class Xavier:
    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2 / (fan_in + fan_out))
        #normally distributed array
        return np.random.normal(0, sigma, weights_shape)

class He:
    def initialize(self, weights_shape,fan_in, fan_out):
        sigma = np.sqrt(2/fan_in)
        return np.random.normal(0, sigma, weights_shape)
