from Layers.Base import BaseLayer

class ReLU(BaseLayer):
    def __init__(self):
        super(ReLU, self).__init__()
        self.trainable = False

    def forward(self, input_tensor):
        # (0,max)
        # print(input_tensor)
        input_tensor[input_tensor <= 0] = 0

        x = input_tensor.copy()

        # if x <= 0, output is 0. if x > 0, output is 1
        # https://stackoverflow.com/questions/32546020/neural-network-backpropagation-with-relu
        # https://datascience.stackexchange.com/questions/19272/deep-neural-network-backpropogation-with-relu

        x[x > 0] = 1
        self.x = x

        return input_tensor

    def backward(self, error_tensor):
        return error_tensor * self.x

#python NeuralNetworkTests.py TestReLU.test_trainable
#python NeuralNetworkTests.py TestReLU.test_forward
#python NeuralNetworkTests.py TestReLU.test_backward
#python NeuralNetworkTests.py TestReLU.test_gradient
