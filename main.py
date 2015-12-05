import image_pro.data as image
import neural as nn
import genetic as gen 
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
def gen_err(pop):
    err = []
    for net in pop:
        error_hearts = 0.0
        error_laugh = 0.0
        error_sad = 0.0
        error_smile = 0.0

        for row in training_hearts:
            e = net.forward_pass(row)
            error_hearts += 1 - e[0]
        for row in training_laugh:
            e = net.forward_pass(row)
            error_hearts += 1 - e[1]
        for row in training_sad:
            e = net.forward_pass(row)
            error_hearts += 1 - e[2]
        for row in training_smile:
            e = net.forward_pass(row)
            error_hearts += 1 - e[3]

        error_hearts = error_hearts/45.0
        error_laugh = error_laugh/45.0
        error_sad = error_sad/45.0
        error_smile = error_smile/45.0

        err.append(error_hearts+error_laugh+error_sad+error_smile)
    return err

def genetic_train(pop_size, test_len):
    pop = []
    best_net = None

    training_hearts = csv.reader(open(training_hearts.csv))
    training_laugh = csv.reader(open(training_laugh.csv))
    training_sad = csv.reader(open(training_sad.csv))
    training_smile = csv.reader(open(training_smile.csv))

    for x in xrange(pop_size):
        net = nn.init_net() 
        pop.append(net)

    errors = gen_err(pop)
    #sorts the errors array but only stores indices 
    idx_err = sorted(range(len(errors)), key=lambda k: errors[k])
    # put_weights1d(self, weights)
    # get_weights(self)
    # emojis = ["hearts", "laugh", "sad", "smile"]

    # decides best 2 parents based on confidence matrix 
    
    while errors[idx_err[0]] > 0.1:
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

    best_net = pop[idx_err[0]]
    return best_net


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


# net = backprop_train(cfg.TRAIN_LEN)

# save_net(net, "25k.csv")
# # net = load_net("./nets/good.csv")
# training_results(net)

#gen alg
net = genetic_train(cfg.POP_SIZE, cfg.TRAIN_LEN)


