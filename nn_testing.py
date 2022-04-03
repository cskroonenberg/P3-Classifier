from nn_classifier import train_and_test
#First test
learning_rate = 1e-3 # How fast the network will change its weights
hidden1 = 500 # Number of neurons in first hidden layer
hidden2 = 1000 # Number of neurons in second hidden layer
hidden3 = 100 # Number of neurons in third hidden layer
output = 10 # Number of neurons in output layer

#TODO: Make so it can use alternative activation functions, optimization functions (SGD, Adam, etc.) or loss functions.

def trials(trials, learning_list, layer_list):
    """
    Function to begin training a customized neural network and report the results. This can be changed
    to have another neural network or expanded to experiment with various training functions.
    :param trials: Number of trials being done as an int
    :param learning_list: List of different learning rates
    :param layer_list: 4 x n list of the sizes of each layer in the nn.
    The layers are 3 hidden and 1 output.
    :return: Accuracy results, Loss results
    """
    for i in range(trials):
        train_and_test(learning_list[i], layer_list[i][0], 
        layer_list[i][1], layer_list[i][2], layer_list[i][3])

def main():
    #Trial with default values
    trials(1, [learning_rate], [[hidden1, hidden2, hidden3, output]])

main()