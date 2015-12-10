import neural as nn
import helpers as helpers
import config as cfg
import sys
import getopt
import imageConverter as converter
import os
import genetic as gen


# Main function that starts the program and handles user input
def main(argv):
    net_name = ""
    train_len = 500
    alg = ""
    autosave_name = ""
    testing_name = ""

    # read command line arguments
    try:
        opts, args = getopt.getopt(argv, "hn:l:c:a:as:t:")
    except getopt.GetoptError:
        print ("Usage: main.py -n <net file to use> -l <training length>, -c <convert images again>, -a <algorithm to use (GEN, BP)> -s <name for autosaving net> -t <name for auto testing file>")
        sys.exit(2)
    if not argv:
        print ("Usage: main.py -n <net file to use> -l <training length>, -c <convert images again>, -a <algorithm to use (GEN, BP)> -s <name for autosaving net> -t <name for auto testing file>")
        sys.exit(2)
    for opt, arg in opts:

        # print usage
        if opt == '-h':
            print ("Usage: main.py -n <net file to use> -l <training length>, -c <convert images again>, -a <algorithm to use (GEN, BP)> -s <name for autosaving net> -t <name for auto testing file>")
            sys.exit()

        # store the name of the net to be loaded
        elif opt == "-n":
            net_name = arg

        # store training length
        elif opt == "-l":
            train_len = int(arg)

        # convert images
        elif opt == "-c":
            converter.convert_images()

        # store the name for the net to be saved
        elif opt == "-s":
            if arg != "":
                autosave_name = arg
            else:
                print("Invalid input for -s")

        # store the algorithm to use
        elif opt == "-a":
            print(arg)
            if arg == "GEN" or arg == "BP":
                alg = arg
            else:
                print ("Non-valid algorithm. Choose either GEN for genetic algorithm or BP for Backpropagation")
                sys.exit()
        elif opt == "-t":
            testing_name = arg


    # initialize net
    # if a net name was given, load that net
    if net_name != "":
        net = helpers.load_net(net_name)

    elif alg == "":
        print ("Neither a previous net or training algorithm was chosen. Try again!")
        sys.exit()

    elif alg == "GEN":

        # train the net using gen alg
        net = gen.genetic_train(cfg.POP_SIZE, train_len, testing_name)

        # if autosave name was given, save the net
        if autosave_name != "":
            helpers.save_net(net, autosave_name)
    else:

        # train the net using BP
        net = nn.backprop_train(train_len, testing_name)

         # if autosave name was given, save the net
        if autosave_name != "":
            helpers.save_net(net, autosave_name)

    print("Net ready!")

    # Ask the user for a command
    while True:
        training_data = []
        print("Choose an action:")
        print("Q - Quit")
        print("T - Run training data")
        print("F - Test a file")
        print("S - Save the net")
        print("C - Continue training")
        print("E - Calculate Error")
        choice = raw_input("")

        # Exit
        if choice in ("Q", "q"):
            print("Exiting!")
            sys.exit()

        # run the training results
        elif choice in ("T", "t"):
            helpers.training_results(net)

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
                helpers.test_image(test_file, net)
        elif choice in ("S", "s"):

            # Save the net
            name = ""
            while name == "":
                name = raw_input("filename: ")
            helpers.save_net(net, name)
        elif choice in ("C", "c"):

            # prompt the user for parameters
            train_len = raw_input("training length? ")
            while not train_len.isdigit():
                train_len = raw_input("try again! ")
            train_len = int(train_len)
            alg = raw_input("Which algorithm to use? GEN/BP: ")
            while not (alg == "GEN" or alg == "BP"):
                alg = raw_input("Try again! GEN/BP: ")

            # train the net
            if alg == "GEN":
                gen.continue_gen(net, train_len, cfg.POP_SIZE)
            else:
                nn.continue_bp(net, train_len)
        elif choice in ("E", "e"):

            # if training data is not already loaded, load it
            if not training_data:
                training_data = helpers.load_training_data()
            print("Calculating...")

            # print out the total error over training data
            print("Total error for net: " + str(gen.net_error(net, training_data)))




if __name__ == "__main__":
    main(sys.argv[1:])


