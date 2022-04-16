from pickletools import optimize
from nn_classifier import train_and_test
from nn_plots import plot_loss
#First test

hidden1 = 500 # Number of neurons in first hidden layer
hidden2 = 1000 # Number of neurons in second hidden layer
hidden3 = 100 # Number of neurons in third hidden layer
output = 10 # Number of neurons in output layer

#TODO: Make so it can use alternative activation functions, optimization functions (SGD, Adam, etc.) or loss functions.

def trials(name, learning_list, layer_list, optimizers, n):
    """
    Function to begin training a customized neural network and report the results. This can be changed
    to have another neural network or expanded to experiment with various training functions.
    "param name: Name of trial as str
    :param trials: Number of trials being done as an int
    :param learning_list: List of different learning rates (how fast the network will change its weights)
    :param layer_list: 4 x n list of the sizes of each layer in the nn.
    :param optimizers: Optimization function as a list of str.
    The layers are 3 hidden and 1 output.
    :return: Accuracy results, Loss results
    """
    if(len(layer_list) != len(learning_list)):
        raise Exception("Learning list size does not match nn list size.")    

    trials = len(layer_list)
    loss_collection = []
    convergence_collection = []
    for i in range(trials):
        print(f"---------------Test {i+1}---------------")
        print(f"             {name}\n")
        accuracy, loss_data, convergence_point = train_and_test(learning_list[i], layer_list[i][0], 
        layer_list[i][1], layer_list[i][2], layer_list[i][3], optim=optimizers[i], n=n[i])
        print("\n\n\n")
        loss_collection.append(loss_data)
        convergence_collection.append(convergence_point)

    return loss_collection, convergence_collection

def tests():
    """
    """
    #Trial with default values

    #Original trial
    t1_layers = [hidden1, hidden2, hidden3, output]
    #loss, convergence = trials("Original",[1e-3], [t1_layers], ["adam"])
    #plot_loss(loss, convergence, ["original"], "Original")
    #print(convergence)
    
    #Trial 1: Learning rate changes (.1e-3 had best results)
    loss, convergence = trials("Learning rate", [.1e-3, 1e-3, 2e-3, 3e-3, 5e-3], [t1_layers] * 5, ["adam"] * 5, [250] * 5)
    plot_loss(loss, convergence, [".1e-3", "1e-3", "2e-3", "3e-3","5e-3"], "learning rate")
    
    #Trial 2: NN size change #Could run this over several iterations
    loss_size, convergence_size = trials("Size", [.1e-3] * 5, [[500, 250, 50, output], 
    [1000, 5000, 1000, 30],[1000, 500, 500, 5],[hidden1, hidden2, hidden3, output,1], [hidden1, hidden2, hidden3, output,10]], ["adam"] * 5, [250] * 5) 
    plot_loss(loss_size, convergence_size, ["Small", "Large", "Medium", "Long", "15 layer"], "NN Size")

    #Trial 3: Differing Optimization functions : #3 tests
    loss_size, convergence_size = trials("Optimizer", [.1e-3] * 2, [t1_layers] * 2, ["adam", "sgd"], [250] * 2) #lgbfs needs closure 
    plot_loss(loss_size, convergence_size, ["Adam", "SGD"], "Optim")
    
    #Trial 4: Differing training iterations
    loss_size, convergence_size = trials("Iterations", [.1e-3] * 5, [t1_layers] * 5, ["adam"] * 5, [100, 250, 500, 750, 1000]) #lgbfs needs closure 
    plot_loss(loss_size, convergence_size, ["100", "250", "500", "750", "1000"], "Iterations")

def main():
    tests()

main()