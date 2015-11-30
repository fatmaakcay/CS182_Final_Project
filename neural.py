import neural_net.data as nn

__author__ = 'Timothy H. C. Tamm'

# num_inputs = 40**2
# num_outputs = 4
# num_hidden_layers = 1
# num_npl = 40**2

# THIS IS TESTING DATA!
num_inputs = 2
num_outputs = 2
num_hidden_layers = 1
num_npl = 2
biases = [.35, .60]

net = nn.NeuralNet(num_inputs, num_outputs, num_hidden_layers, num_npl, biases)

training_data = [[0.05, 0.1], [0.01, 0.99]]
net.put_weights([[[.15, .25], [.20, .30]], [[.4, .5], [.45, .55]]])

print("first outs: " + str(net.forward_pass(training_data[0])))
for _ in range(100):
    net.train(training_data[0], training_data[1])
print("new outs: " + str(net.forward_pass(training_data[0])))
