import image_pro.data as image
import genetic as genetic
import neural as nn
import glob as glob
import random

# possible outputs
emojis = ["disgust", "laugh", "sad", "smile"]
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
    for _ in range(test_len):
        # randomly pick an emotion to train
        emotion = random.choice(range(len(emojis)))
        # pick a specific test case
        test_cases = glob.glob('./test_data/' + emojis[emotion] + '/*.png')
        inputs = image.convert_to_1d(image.binary_image(test_cases))
        net.train(inputs, outputs[emotion])
    return net


# TODO: Function for training neural net with genetic algorithm
def genetic_train():
    net = nn.init_net()
    return net
