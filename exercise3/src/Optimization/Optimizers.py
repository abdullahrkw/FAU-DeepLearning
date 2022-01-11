"""optimizers
"""
import numpy as np


class Sgd:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def calculate_update(self,weight_tensor, gradient_tensor):
        weight_tensor = weight_tensor- self.learning_rate*gradient_tensor
        return weight_tensor

class SgdWithMomentum:
    def __init__(self, learing_rate, momentum_rate):
        self.learing_rate = learing_rate
        self.momentum_rate = momentum_rate
        self.v = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.v = self.momentum_rate * self.v - self.learing_rate * gradient_tensor
        updated_weight = weight_tensor + self.v
        return updated_weight

class Adam:
    def __init__(self, learing_rate, mu, rho):
        self.learing_rate = learing_rate
        self.mu = mu
        self.rho = rho
        self.v = 0
        self.r = 0
        self.k = 0

    def calculate_update(self, weight_tensor, gradient_tensor):

        g = gradient_tensor
        mu = self.mu
        rho = self.rho
        self.k += 1
        self.v = mu * self.v + (1 - mu) * g
        self.r = rho * self.r + (1 - rho) * g * g

        v_hat = self.v / (1 - np.power(mu, self.k))
        r_hat = self.r / (1 - np.power(rho, self.k))

        updated_weight = weight_tensor - self.learing_rate * (v_hat) / (np.sqrt(r_hat) + np.finfo(float).eps)
        return updated_weight
