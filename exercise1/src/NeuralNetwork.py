import copy
import numpy as np 

from Layers.Base import BaseLayer


class NeuralNetwork(object):
    def __init__(self, optimizer=None):
        self.optimizer = optimizer
        self.loss = list()
        self.layers = list()
        self.data_layer = None
        self.loss_layer = None
        self._inpT = None
        self._labelT = None

    def _get_data(self) -> tuple:
        self._inpT, self._labelT = self.data_layer.next()
        return self._inpT, self._labelT

    def append_layer(self, layer:BaseLayer) -> None:
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)

    def forward(self) -> float:
        inpT, self._labelT = self._get_data()
        for layer in self.layers:
            inpT = layer.forward(inpT)
        return self.loss_layer.forward(inpT, self._labelT)

    def backward(self) -> None:
        errT = self.loss_layer.backward(self._labelT)
        for layer in reversed(self.layers):
            errT = layer.backward(errT)

    def train(self, itrs) -> None:
        if not isinstance(itrs, int):
            raise ValueError(f"{itrs} is not an int")
        for itr in range(itrs):
            loss = self.forward()
            self.loss.append(loss)
            self.backward()

    def test(self, inpT) -> np.ndarray:
        for layer in self.layers:
            inpT = layer.forward(inpT)
        return inpT
