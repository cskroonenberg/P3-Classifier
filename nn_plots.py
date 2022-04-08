from lib2to3.pgen2.token import PLUS
from re import M
import matplotlib.pyplot as plt
from pylab import rcParams

def plot_loss(loss_collection, convergence_points, labels, title):
    """
    Function plots loss data with a line chart
    :param loss_collection: Loss data with different experiments
    """
    rcParams['figure.figsize'] = 10, 5
    for i in range(len(loss_collection)):
        plt.plot(list(range(0, len(loss_collection[i]))), loss_collection[i], label=labels[i])
        plt.scatter(convergence_points[i][0],convergence_points[i][1],marker="+", color="black")
    plt.title(f"Loss Vs Iterations with {title}")
    plt.ylabel("Loss", fontsize=15)
    plt.xlabel("Iteration")
    plt.legend()
    plt.show()