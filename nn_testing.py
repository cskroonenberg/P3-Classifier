from nn_classifier import train_and_test
from nn_plots import plot_loss
#First test

hidden1 = 500 # Number of neurons in first hidden layer
hidden2 = 1000 # Number of neurons in second hidden layer
hidden3 = 100 # Number of neurons in third hidden layer
output = 10 # Number of neurons in output layer

#TODO: Make so it can use alternative activation functions, optimization functions (SGD, Adam, etc.) or loss functions.

def trials(name, trials, learning_list, layer_list):
    """
    Function to begin training a customized neural network and report the results. This can be changed
    to have another neural network or expanded to experiment with various training functions.
    "param name: Name of trial as str
    :param trials: Number of trials being done as an int
    :param learning_list: List of different learning rates (how fast the network will change its weights)
    :param layer_list: 4 x n list of the sizes of each layer in the nn.
    The layers are 3 hidden and 1 output.
    :return: Accuracy results, Loss results
    """
    loss_collection = []
    for i in range(trials):
        print(f"---------------Test {i+1}---------------")
        print(f"             {name}\n")
        _,_, accuracy, loss_data, convergence_point = train_and_test(learning_list[i], layer_list[i][0], 
        layer_list[i][1], layer_list[i][2], layer_list[i][3])
        print("\n\n\n")
        loss_collection.append(loss_data)

    return loss_collection

def tests():
    """
    """
    #Trial with default values

    #Original trial
    t1_layers = [hidden1, hidden2, hidden3, output]
    loss = trials("Original",1,[1e-3], [t1_layers])

    #Trial 1: Learning rate changes (2e-3 had best results)
    #loss = trials("Learning rate",5, [.1e-3, 1e-3, 2e-3, 3e-3, 5e-3], [t1_layers, t1_layers,t1_layers,t1_layers,t1_layers])
    #plot_loss(loss, [".1e-3", "1e-3", "2e-3", "3e-3","5e-3"], "learning rate")
    #Trial 2: NN size change #Could run this over several iterations
    #loss_size = trials("Size", 3, [2e-3,2e-3,2e-3], [[500, 250, 50, output], [1000, 5000, 1000, 30],[1000, 500, 500, 5]]) 
    #plot_loss(loss_size, ["Small", "Large", "Medium"], "NN Size")

def main():
    tests()

main()