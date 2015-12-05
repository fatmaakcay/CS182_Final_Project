import neural_net.data as nn
import config as cfg
import helpers
import random

__author__ = 'Timothy H. C. Tamm'


def init_net():

    num_inputs = cfg.RES[0]*cfg.RES[1]
    num_outputs = 4
    num_hidden_layers = 1
    num_npl = cfg.NPL

    return nn.NeuralNet(num_inputs, num_outputs, num_hidden_layers, num_npl)

# function for training neural net with backpropagation
def backprop_train(test_len):

    # load training data
    training_inputs = helpers.load_training_data()

    # initalize a net
    net = nn.init_net()

    # train the net
    for i in range(test_len):

        # Print out progress:
        if i % 100 == 0:
            per = float(i) / float(test_len) * 100
            print("Training: " + str(format(per, '.2f')) + "%")

        # randomly pick an emotion to train
        emotion = random.choice(range(len(cfg.emojis)))

        # pick a specific test case
        test_case = random.choice(training_inputs[emotion])

        net.train(test_case, cfg.outputs[emotion])
    return net


def error(ideal, actual):
    return nn.total_error(ideal, actual)
