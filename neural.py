__author__ = 'Timothy H. C. Tamm'

import neural_net.data as NN

num_inputs = 40**2
num_outputs = 4
num_hidden_layers = 2
num_npl = 40**2
bias = 0

net = NN.NeuralNet(num_inputs, num_outputs, num_hidden_layers, num_npl, bias)
