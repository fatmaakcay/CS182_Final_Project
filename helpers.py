import csv
import os
import sys
import neural as nn
import image_pro.data as image
import config as cfg
import glob

__author__ = 'timothy'

# Normalizes the results into a confidence matrix
def normalize(array):
    sum = 0
    for n in array:
        sum += n
    for i in range(len(array)):
        array[i] = array[i]/sum
    return array

# Loads previously converted training data
def load_training_data():

    # Check that training data has been converted:
    if not os.access("./test_data/hearts/converted.csv", os.R_OK) or \
            not os.access("./test_data/laugh/converted.csv", os.R_OK) or \
            not os.access("./test_data/sad/converted.csv", os.R_OK) or \
            not os.access("./test_data/smile/converted.csv", os.R_OK):
        print("training data can't be accessed. Make sure it's converted")
        sys.exit()

    #initialize training data
    readers = [
        csv.reader(open("./test_data/hearts/converted.csv")),
        csv.reader(open("./test_data/laugh/converted.csv")),
        csv.reader(open("./test_data/sad/converted.csv")),
        csv.reader(open("./test_data/smile/converted.csv")),
    ]
    training_inputs = [[], [], [], []]

    for i in range(4):
        for row in readers[i]:
            inp = []
            for el in row:
                if el == "0":
                    inp.append(0)
                else:
                    inp.append(1)
            training_inputs[i].append(inp)

    return training_inputs

# run an image through the net and pretty print the results
def test_image(path, net):

    # Run the image
    inp = image.convert_to_1d(image.binary_image(path, cfg.RES))
    result = net.forward_pass(inp)
    normalize(result)

    # print out result
    print(path + ":")
    print("     hearth eyes:" + str(format((result[0]*100), '.2f')) + "%")
    print("     laugh:" + str(format((result[1]*100), '.2f')) + "%")
    print("     sad:" + str(format((result[2]*100), '.2f')) + "%")
    print("     smile:" + str(format((result[3]*100), '.2f')) + "%")

# test net with preset testing data
def training_results(net):

    # choose an emotion:
    for emotion in cfg.emojis:
        # go through the test cases
        test_cases = glob.glob('./test_data/non-training/' + emotion + '*.png')
        for test_case in test_cases:
            # run the test case through the net
            test_image(test_case, net)


def save_net(net, name):

    # choose an emotion
    path = "./nets/" + name
    data = net.get_weights()
    with open(path, 'wb+') as csv_file:
        weight_writer = csv.writer(csv_file, delimiter=',')
        weight_writer.writerow(data)


def load_net(name):
    print("Loading net...")
    path = "./nets/" + name
    input = open(path, "r")
    net = nn.init_net()
    weights = []
    with open(path, 'rb') as csv_file:
        weight_reader = csv.reader(csv_file, delimiter=',')
        for row in weight_reader:
            for el in row:
                weights.append(float(el))
    net.put_weights1d(weights)
    print("Net loaded!")
    return net