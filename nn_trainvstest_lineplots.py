import matplotlib.pyplot as plt

def plotline_trainvstest(f1_collection, labels, title):
    """
    Function plots train v test data
    :param f1_collection: Loss data with different experiments
    :param labels: Names for each entry as str
    :param title: Name of the trial as str
    @return: N/A
    """
    xpos = list(range(0, (len(f1_collection))))
    training_scores = [item[0] for item in f1_collection]
    testing_scores = [item[1] for item in f1_collection]
    plt.plot(training_scores, label="Training")
    plt.plot(testing_scores, label="Testing")
    
    plt.title(f"Training vs Test Scores with: {title}")
    plt.ylabel("F1-Score", fontsize=14)
    plt.ylim(0.6, 1.05)
    plt.xticks(xpos, labels)
    #plt.xlabel("", fontsize=15)
    plt.legend()
    plt.show()