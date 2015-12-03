import neural_net.data as nn

__author__ = 'Timothy H. C. Tamm'


def init_net():

    num_inputs = 40**2
    num_outputs = 4
    num_hidden_layers = 1
    num_npl = 40**2

    return nn.NeuralNet(num_inputs, num_outputs, num_hidden_layers, num_npl)
