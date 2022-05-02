import matplotlib.pyplot as plt

def plot_trainvstest(f1_collection, labels, title):
    """
    Function plots train v test data
    :param f1_collection: Loss data with different experiments
    :param labels: Names for each entry as str
    :param title: Name of the trial as str
    @return: N/A
    """
    xpos = list(range(1, (len(f1_collection) + 1)))
    train_xpos = [x-0.1 for x in xpos]
    test_xpos = [x+0.2 for x in xpos]
    training_scores = [item[0] for item in f1_collection]
    testing_scores = [item[1] for item in f1_collection]
    plt.bar(train_xpos, training_scores, width=0.3, label="Training")
    plt.bar(test_xpos, testing_scores, width=0.3, label="Testing")
    
    plt.xticks(xpos, labels)
    plt.title(f"Training vs Test Scores with: {title}")
    plt.ylabel("F1-Score", fontsize=14)
    plt.ylim(0.6, 1.05)
    #plt.xlabel("", fontsize=15)
    plt.legend()
    plt.show()