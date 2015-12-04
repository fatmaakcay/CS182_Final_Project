import neural_net.data as nn
import config as cfg

__author__ = 'Timothy H. C. Tamm'


def init_net():

    num_inputs = cfg.RES[0]*cfg.RES[1]
    num_outputs = 4
    num_hidden_layers = 1
    num_npl = cfg.NPL**2

    return nn.NeuralNet(num_inputs, num_outputs, num_hidden_layers, num_npl)
