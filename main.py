import neural as nn
import helpers as helpers
import config as cfg
import sys
import getopt
import imageConverter as converter
import os
import genetic as gen


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
        net = helpers.load_net(net_name)
    elif alg == "":
        print ("Neither a previous net or training algorithm was chosen. Try again!")
        sys.exit()
    elif alg == "GEN":
        net = gen.genetic_train(cfg.POP_SIZE, train_len)
    else:
        net = nn.backprop_train(train_len)

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



if __name__ == "__main__":
    main(sys.argv[1:])


