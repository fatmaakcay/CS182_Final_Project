import neural as nn
import config as cfg
import csv
import helpers as helpers
import genetic_alg.genetic as gen


# Error function for genetic alg
def gen_err(pop, training_data):

    err = []
    for net in pop:
        err.append(net_error(net, training_data))
    return err


# Training data error for one net
def net_error(net, training_data):
    error = [0.0, 0.0, 0.0, 0.0]
    for i in range(4):
        for row in training_data[i]:
            e = net.forward_pass(row)
            error[i] += nn.error(cfg.outputs[i], e)
        error[i] /= len(training_data[i])
    return sum(error)


# Function for training neural net with genetic algorithm
def genetic_train(pop_size, test_len, testing_name):

    writer = None
    csv_file = None
    if testing_name != "":
        path = "./results/" + testing_name
        csv_file = open(path, 'wb+')
        writer = csv.writer(csv_file, delimiter=',')

    print("Training the net!")
    # Initializes the first population
    print("Generation: 0")
    pop = []
    for x in xrange(pop_size):
        net = nn.init_net()
        pop.append(net)

    training_data = helpers.load_training_data()

    # Calculate errors
    errors = gen_err(pop, training_data)

    # Sorts the errors array but only stores indices
    idx_err = sorted(range(len(errors)), key=lambda k: errors[k])

    print(" Smallest error: " + str(errors[idx_err[0]]))

    if writer is not None:
        data = [0, helpers.get_testing_error(pop[idx_err[0]])]
        writer.writerow(data)

    # Goes through the generations
    counter = 1
    while errors[idx_err[0]] > 0.1 and counter < test_len:

        print("Generation: " + str(counter))

        # Decides best 2 parents based on confidence matrix
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
        print(" Smallest error: " + str(errors[idx_err[0]]))

        # write the current test_error
        if writer is not None and counter % 5 == 0:
            data = [counter, helpers.get_testing_error(pop[idx_err[0]])]
            writer.writerow(data)
        counter += 1

    # write the last test error:
    if writer is not None:
        data = [counter, helpers.get_testing_error(pop[idx_err[0]])]
        writer.writerow(data)

    # close the file
    if csv_file is not None:
        csv_file.close()

    best_net = pop[idx_err[0]]
    return best_net


def continue_gen(net, train_len, pop_size):

    w1 = net.get_weights()
    mut_cnt = int(cfg.MUT_RATE * len(w1))
    print("Training the net!")
    # Initializes the first population
    print("Generation: 0")
    pop = []
    for x in xrange(pop_size):
        new_net = nn.init_net()
        weights = gen.mutate(w1, mut_cnt)
        new_net.put_weights1d(weights)
        pop.append(new_net)

    training_data = helpers.load_training_data()

    # Calculate errors
    errors = gen_err(pop, training_data)

    # Sorts the errors array but only stores indices
    idx_err = sorted(range(len(errors)), key=lambda k: errors[k])

    print(" Smallest error: " + str(errors[idx_err[0]]))

    # Goes through the generations
    counter = 1
    while counter < train_len:

        print("Generation: " + str(counter))

        # Decides best 2 parents based on confidence matrix
        parent1 = pop[idx_err[0]]
        parent2 = pop[idx_err[1]]
        w1 = parent1.get_weights()
        w2 = parent2.get_weights()
        for x in idx_err[2:]:
            new_w = gen.recombine(w1, w2)
            new_w = gen.mutate(new_w, mut_cnt)
            pop[x].put_weights1d(new_w)

        errors = gen_err(pop, training_data)
        idx_err = sorted(range(len(errors)), key=lambda k: errors[k])
        print(" Smallest error: " + str(errors[idx_err[0]]))
        counter += 1

    best_net = pop[idx_err[0]]
    return best_net

