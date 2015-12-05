import neural as nn
import config as cfg
import helpers as helpers
import genetic_alg.genetic as gen

# Error function for genetic alg
def gen_err(pop, training_data):

    err = []
    for net in pop:
        error_hearts = 0.0
        error_laugh = 0.0
        error_sad = 0.0
        error_smile = 0.0

        for row in training_data[0]:
            e = net.forward_pass(row)
            error_hearts += nn.error(cfg.outputs[0], e)
        for row in training_data[1]:
            e = net.forward_pass(row)
            error_laugh += nn.error(cfg.outputs[1], e)
        for row in training_data[2]:
            e = net.forward_pass(row)
            error_sad += nn.error(cfg.outputs[2], e)
        for row in training_data[3]:
            e = net.forward_pass(row)
            error_smile += nn.error(cfg.outputs[3], e)

        error_hearts /= len(training_data[0])
        error_laugh /= len(training_data[1])
        error_sad /= len(training_data[2])
        error_smile /= len(training_data[3])

        err.append(error_hearts+error_laugh+error_sad+error_smile)

    return err


# Function for training neural net with genetic algorithm
def genetic_train(pop_size, test_len):

    print("Training the net!")
    # initialize the first population
    print("Generation: 0")
    pop = []
    for x in xrange(pop_size):
        net = nn.init_net()
        pop.append(net)

    training_data = helpers.load_training_data()

    # Calculate erros
    errors = gen_err(pop, training_data)

    # sorts the errors array but only stores indices
    idx_err = sorted(range(len(errors)), key=lambda k: errors[k])

    # go through the generations
    counter = 1
    while errors[idx_err[0]] > 0.1 or counter > test_len:

        print("Generation: " + str(counter))

        # decides best 2 parents based on confidence matrix
        parent1 = pop[idx_err[0]]
        parent2 = pop[idx_err[1]]
        w1 = parent1.get_weights()
        w2 = parent2.get_weights()
        mut_cnt = int(cfg.MUT_RATE * len(w1))
        for x in idx_err[2:]:
            new_w = gen.recombine(w1, w2)
            new_w = gen.mutate(new_w, mut_cnt)
            pop[x].put_weights1d(new_w)

        errors = gen_err(pop, training_data)
        idx_err = sorted(range(len(errors)), key=lambda k: errors[k])
        counter += 1

    best_net = pop[idx_err[0]]
    return best_net