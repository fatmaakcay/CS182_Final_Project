import image_pro.data as image
import neural as nn
import genetic as gen 
import glob as glob
import random
import config as cfg
import csv
import sys
import getopt
import imageConverter as converter
import os

# Categories
emojis = ["hearts", "laugh", "sad", "smile"]

# Ideal outputs for each category
outputs = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
]


# Normalizes the results into a confidence matrix
def normalize(array):
    sum = 0
    for n in array:
        sum += n
    for i in range(len(array)):
        array[i] = array[i]/sum
    return array


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

# function for training neural net with backpropagation
def backprop_train(test_len):

    # load training data
    training_inputs = load_training_data()

    # initalize a net
    net = nn.init_net()

    # train the net
    for i in range(test_len):

        # Print out progress:
        if i % 100 == 0:
            per = float(i) / float(test_len) * 100
            print("Training: " + str(format(per, '.2f')) + "%")

        # randomly pick an emotion to train
        emotion = random.choice(range(4))

        # pick a specific test case
        test_case = random.choice(training_inputs[emotion])

        net.train(test_case, outputs[emotion])
    return net


# Error function for genetic alg
def gen_err(pop):

    training_data = load_training_data()

    err = []
    for net in pop:
        error_hearts = 0.0
        error_laugh = 0.0
        error_sad = 0.0
        error_smile = 0.0

        for row in training_data[0]:
            e = net.forward_pass(row)
            error_hearts += 1 - e[0]
        for row in training_data[1]:
            e = net.forward_pass(row)
            error_hearts += 1 - e[1]
        for row in training_data[2]:
            e = net.forward_pass(row)
            error_hearts += 1 - e[2]
        for row in training_data[3]:
            e = net.forward_pass(row)
            error_hearts += 1 - e[3]

        error_hearts /= len(training_data[0])
        error_laugh /= len(training_data[1])
        error_sad /= len(training_data[2])
        error_smile /= len(training_data[3])

        err.append(error_hearts+error_laugh+error_sad+error_smile)

    return err


# Function for training neural net with genetic algorithm
def genetic_train(pop_size, test_len):

    # initialize the first population
    pop = []
    for x in xrange(pop_size):
        net = nn.init_net() 
        pop.append(net)

    # Calculate erros
    errors = gen_err(pop)

    # sorts the errors array but only stores indices
    idx_err = sorted(range(len(errors)), key=lambda k: errors[k])

    # go through the generations
    counter = 0
    while errors[idx_err[0]] > 0.1 or counter > test_len:

        print("Generation: " + str(counter))

        # decides best 2 parents based on confidence matrix
        parent1 = pop[idx_err[0]]
        parent2 = pop[idx_err[1]]
        w1 = parent1.get_weights()
        w2 = parent2.get_weights()
        for x in idx_err[2:]:
            new_w = gen.recombine(w1, w2)
            new_w = gen.mutate(new_w)
            pop[x].put_weights1d(new_w)

        errors = gen_err(pop)
        idx_err = sorted(range(len(errors)), key=lambda k: errors[k])
        counter += 1

    best_net = pop[idx_err[0]]
    return best_net


def training_results(net):

    # choose an emotion:
    for emotion in emojis:
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


def main(argv):
    net_name = ""
    train_len = 500
    alg = ""
    try:
        opts, args = getopt.getopt(argv, "hn:l:c:a:")
    except getopt.GetoptError:
        print ("Usage: main.py -n <net file to use> -l <training length>, -c <convert images again>, -a <algorithm to use (GEN, BP)>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ("Usage: main.py -n <net file to use> -l <training length>, -c <convert images again (y, n)>, -a <algorithm to use (GEN, BP)>")
            sys.exit()
        elif opt == "-n":
            net_name = arg
        elif opt == "-l":
            train_len = arg
        elif opt == "-c":
            converter.convert_images()
        elif opt == "-a":
            if arg == "GEN" or arg == "BP":
                alg = arg
            else:
                print ("Non-valid algorithm. Choose either GEN for genetic algorithm or BP for Backpropagation")
                sys.exit()

    # initialize net
    if net_name != "":
        net = load_net(net_name)
    elif alg == "":
        print ("Neither a previous net or training algorithm was chosen. Try again!")
        sys.exit()
    elif alg == "GEN":
        net = genetic_train(cfg.POP_SIZE, train_len)
    else:
        net = backprop_train(train_len)

    print("Net ready!")

    # Ask the user for a command
    while True:
        print("Choose an action:")
        print("Q - Quit")
        print("T - Run training data")
        print("F - Test a file")
        print("S - Save the net")
        choice = raw_input("")

        # Exit
        if choice in ("Q", "q"):
            print("Exiting!")
            sys.exit()

        # run the training results
        elif choice in ("T", "t"):
            training_results(net)

        # test a specific file
        elif choice in ("F", "f"):

            # get file path
            test_file = raw_input("input path to file: ")
            while test_file == "":
                test_file = raw_input("Try again!")

            # Check if file is accessible
            if not os.access(test_file, os.R_OK):
                print("file is not accessible :( ")
            else:
                test_image(test_file, net)
        elif choice in ("S", "s"):
            # Save the net
            name = ""
            while name == "":
                name = raw_input("filename: ")
            save_net(net, name)



if __name__ == "__main__":
    main(sys.argv[1:])


