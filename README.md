<!-- # CS182_Final_Project

Our system relies on the Python Imaging Library (PIL). In order to use our system you first need to download the library, which is available for free here: http://www.pythonware.com/products/pil/ \\

The system is operated using a combination of command line arguments and a terminal based UI.\\
To run the program you need to execute main.py with the following arguments:
-n $\langle$name of net$\rangle$ - used for loading a neural net that has been stored in /nets/ \\
-l $\langle$training length$\rangle$ - number of iterations/generations to run through when training the net \\
-h - help. Shows usage\\
-c - Converts the training/testing images again. Use only if you have added/removed images \\
-a $\langle$BP/GEN$\rangle$ - Which algorithm to use for training. BP - Backpropagation, GEN - genetic algorithm. \\
-s $\langle$filename$\rangle$ - Used for automatically saving a trained neural net as filename.csv in /nets/ \\
-t $\langle$filename$\rangle$ - Used for automatically and periodically running the testing data through the net and saving the result in a csv file with the provided filename in /results/ while the net is being trained.\\ \\

All arguments are optional, but either -n or -a must be filled. If both are filled, then the system will ignore -a and just load the saved net.\\
Once the system has either loaded or trained a net it will prompt the user for an action from the following list:\\
Q - Quit\\
T - Run training data\\
F - Test a file\\
S - Save the net\\
C - Continue training\\
E - Calculate Error\\ \\
Q will quit the system.\\
T will run the pictures stored in $/$test\_data$/$non-training$/$ through the net and output the results. (the filenames for these pictures have to start with the emotion that they represent).\\
F will prompt the user for the path to a picture file, which it will then convert and run through the net, finally displaying the results.\\
S will prompt the user for a filename and save the current net in /nets/.\\
C will prompt the user for a training length and algorithm and then train the current net.\\
E will calculate the average squared error of the current net using the training data and outputs the result.
 -->