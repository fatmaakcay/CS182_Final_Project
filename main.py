import image_pro.data as image
import neural as nn
import glob as glob
import random
import config as cfg
import csv

# possible outputs
emojis = ["hearts", "laugh", "sad", "smile"]
outputs = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
]


def normalize(array):
    sum = 0
    for n in array:
        sum += n
    for i in range(len(array)):
        array[i] = array[i]/sum
    return array

# function for training neural net with backpropagation
def backprop_train(test_len):

    net = nn.init_net()
    # train the net
    for i in range(test_len):
        if i % 100 == 0:
            per = float(i) / float(test_len) * 100
            print("Training: " + str(format(per, '.2f')) + "%")
        # randomly pick an emotion to train
        emotion = random.choice(range(len(emojis)))
        # pick a specific test case
        test_cases = glob.glob('./test_data/' + emojis[emotion] + '/*.png')
        test_case = random.choice(test_cases)
        inputs = image.convert_to_1d(image.binary_image(test_case, cfg.RES))
        net.train(inputs, outputs[emotion])
    return net


# TODO: Function for training neural net with genetic algorithm
def genetic_train():
    net = nn.init_net()
    return net


def training_results(net):

    # choose an emotion:
    for emotion in emojis:
        # go through the test cases
        test_cases = glob.glob('./test_data/non-training/' + emotion + '*.png')
        for test_case in test_cases:
            # run the test case through the net
            inp = image.convert_to_1d(image.binary_image(test_case, cfg.RES))
            result = net.forward_pass(inp)
            normalize(result)
            # print out result
            print(emotion + ":")
            print("     hearth eyes:" + str(format((result[0]*100), '.2f')) + "%")
            print("     laugh:" + str(format((result[1]*100), '.2f')) + "%")
            print("     sad:" + str(format((result[2]*100), '.2f')) + "%")
            print("     smile:" + str(format((result[3]*100), '.2f')) + "%")


def save_net(net, name):

    # choose an emotion
    data = net.get_weights()
    path = './nets/' + name
    with open(path, 'wb+') as csv_file:
        weight_writer = csv.writer(csv_file, delimiter=',')
        weight_writer.writerow(data)


def load_net(path):
    print("Loading net...")
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


net = backprop_train(cfg.TRAIN_LEN)
save_net(net, "25k.csv")
# net = load_net("./nets/good.csv")
training_results(net)


