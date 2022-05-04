import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, plot_confusion_matrix

def plot_confusionmtrx(test_results, labels, title):
    """
    Plots multiple matrices
    :param test_results: Predicted and actual results as 2d list
    :param labels: Names for each entry as str
    :param title: Name of the trial as str
    @return: N/A
    """
    for i in range(len(test_results)):
        ConfusionMatrixDisplay.from_predictions(test_results[i][1], test_results[i][0],
        cmap=plt.cm.Blues)
    
        plt.title(f"{title} Confusion Matrix: {labels[i]}")
        plt.ylabel("Actual Label", fontsize=14)
        plt.xlabel("Predicition", fontsize=15)

    plt.show()
                