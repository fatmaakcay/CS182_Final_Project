__author__ = 'Timothy H. C. Tamm'

import random
import math

random.seed()


# function for the NeuralNet output values
def sigmoid(x):
    return math.tanh(x)

# Class of a neuron in the net.
class Neuron(object):
    def __init__(self, num_inputs):
        self.num_inputs = num_inputs
        self.weights = []
        for _ in range(num_inputs + 1):
            self.weights.append(random.random())

    # sums the inputs weighted with the weights.
    def sum(self, inputs):
        S = 0.0
        for i, val in enumerate(inputs):
            S += val*self.weights[i]
        return S

    # given an array of weights, sets them as the Neuron's weights.
    def set_weights(self, weights):
        self.weights = weights


# Neuron Layer class. Holds all the neurons in that layer.
class NeuronLayer(object):
    def __init__(self, num_neurons, num_inputs):
        self.num_neurons = num_neurons
        self.neurons = []
        for _ in range(num_neurons):
            self.neurons.append(Neuron(num_inputs))


# Neural net class. holds all the neuron layers.
class NeuralNet(object):
    def __init__(self, num_inputs, num_outputs, num_hidden_layers, num_neurons_per_hidden_layer, bias):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hidden_layers = num_hidden_layers
        self.num_neurons_per_hidden_layer = num_neurons_per_hidden_layer
        self.bias = bias
        if self.num_hidden_layers > 0:
            self.layers = [NeuronLayer(num_neurons_per_hidden_layer, num_inputs)]
            self.layers += [NeuronLayer(num_neurons_per_hidden_layer, num_neurons_per_hidden_layer)
                            for _ in range(num_hidden_layers)]
            self.layers += [NeuronLayer(num_outputs, num_neurons_per_hidden_layer)]
        else:
            self.layers = [NeuronLayer(num_outputs, num_inputs)]

    # function for outputting all the weights in the net as a 1D array.
    def get_weights(self):
        out = []
        for l in self.layers:
            for n in l.neurons:
                for w in n.weights:
                    out.append(w)
        return out

    # outputs the total number of weights in the net
    def num_of_weights(self):
        if self.num_hidden_layers == 0:
            return self.num_outputs
        else:
            return self.num_outputs + (self.num_hidden_layers + 1) * self.num_hidden_layers

    # Given a 2D array of weights, starting with the first neuron in the first layer and
    # ending with the last neuron in the last array, sets the weights of the all the neurons in the net.
    def put_weights(self, weights):
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.set_weights(weights[:neuron.num_inputs])
                weights = weights[neuron.num_inputs:]

    # Given an input, calculates the output of the NN.
    def update(self, inputs):
        if self.num_inputs != len(inputs):
            raise ValueError("Invalid input")
        else:
            outputs = []
            for layer in self.layers:
                outputs = []
                for n in layer.neurons:
                    outputs.append(sigmoid(n.sum(inputs) + n.weights[-1] * self.bias))
                inputs = outputs
        return outputs