import neural_net.data as nn
import config as cfg
import csv
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
def backprop_train(test_len, testing_name):

    # initalize a net
    net = init_net()

    return continue_bp(net, test_len, testing_name)


def error(ideal, actual):
    return nn.total_error(ideal, actual)


# function for continuing to train a net.
def continue_bp(net, test_len, testing_name=""):

    # load training data
    training_inputs = helpers.load_training_data()

    # writer for testing data
    csv_file = None
    writer = None
    if testing_name != "":
        path = "./results/" + testing_name
        csv_file = open(path, 'wb+')
        writer = csv.writer(csv_file, delimiter=',')

    # train the net
    for i in range(test_len):

        # Print out progress:
        if i % 100 == 0:
            per = float(i) / float(test_len) * 100
            print("Training: " + str(format(per, '.2f')) + "%")

            # write testing results if testing
            if writer is not None:
                data = [i, helpers.get_testing_error(net)]
                writer.writerow(data)

        # randomly pick an emotion to train
        emotion = random.choice(range(len(cfg.emojis)))

        # pick a specific test case
        test_case = random.choice(training_inputs[emotion])

        net.train(test_case, cfg.outputs[emotion])

    # write testing results if testing
    if writer is not None:
        data = [test_len, helpers.get_testing_error(net)]
        writer.writerow(data)

    # close the file
    if csv_file is not None:
        csv_file.close()

    return net
