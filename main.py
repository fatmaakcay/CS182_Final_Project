import image_pro.data as image
import neural as nn
import glob as glob
import random

# possible outputs
emojis = ["hearts", "laugh", "sad", "smile"]
outputs = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
]

# function for training neural net with backpropagation
def backprop_train(test_len):

    net = nn.init_net()
    # train the net
    for i in range(test_len):
        print("Training: " + str(i))
        # randomly pick an emotion to train
        emotion = random.choice(range(len(emojis)))
        # pick a specific test case
        test_cases = glob.glob('./test_data/' + emojis[emotion] + '/*.png')
        test_case = random.choice(test_cases)
        inputs = image.convert_to_1d(image.binary_image(test_case))
        net.train(inputs, outputs[emotion])
    return net


# TODO: Function for training neural net with genetic algorithm
def genetic_train():
    net = nn.init_net()
    return net


def training_results(train_len, type="BP"):

    if type == "BP" or type == "GEN":
        if type == "BP":
            net = backprop_train(train_len)
        else:
            net = genetic_train()
        for emotion in emojis:
            test_cases = glob.glob('./test_data/' + emotion + '/*.png')
            test_case = random.choice(test_cases)
            inp = image.convert_to_1d(image.binary_image(test_case))
            result = net.forward_pass(inp)
            print(emotion + ":")
            print("     hearth eyes:" + str(result[0]*100) + "%")
            print("     laugh:" + str(result[1]*100) + "%")
            print("     sad:" + str(result[2]*100) + "%")
            print("     smile:" + str(result[3]*100) + "%")
    else:
        print("ERROR: Choose correct training type: BP - Backpropagation, GEN - genetic algorithm")


training_results(1000, "BP")
