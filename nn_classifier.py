from ast import Raise
from collections import OrderedDict
from pylab import rcParams
import torch
import torch.nn as nn
import numpy as np
import mne
from extract_positives import extract_p300

# See http://learn.neurotechedu.com/machinelearning/

def train_and_test(learning_rate, hidden1, hidden2, hidden3, output, extra_layers=0, optim="adam"):
  """
  Function to train and test the neural network with customized parameters.
  :param learning_rate: Customized learning rate for training (backprop) as a float.
  :param hidden1: Size of 1st hidden layer as an int
  :param hidden2: Size of 2nd hidden layer as an int
  :param hidden3: Size of 3rd hidden layer as an int
  :param extra_layers: Number of additional layers to add to the nn
  :param optim: Optimization function to chose as a string
  :return: Averages of testing scores, but matplot lib plots testing results.
  """
  # Set randomizer seed for consistency
  torch.manual_seed(100)
  eeg_sample_count = 240 # Number of samples to train network with
  eeg_sample_length = 226 # Number of datapoints per sample
  number_of_classes = 1 # 1 Output variable w/ 1.0 = 100%, 0.0 = 0% certainty that sample has p300  
  
  ## Network
  model = nn.Sequential()

  # Input layer (Size 226 -> 500)
  model.add_module('Input Linear', nn.Linear(eeg_sample_length, hidden1))
  model.add_module('Input Activation', nn.CELU())

  # Hidden Layer (Size 500 -> 1000)
  model.add_module('Hidden Linear', nn.Linear(hidden1, hidden2))
  model.add_module('Hidden Activation', nn.ReLU())

  if extra_layers > 0:
      for i in range(extra_layers):
        model.add_module(f'Extra Linear {i}', nn.Linear(hidden2, hidden2))
        model.add_module(f'Hidden Activation {i}', nn.ReLU())

  # Hidden Layer (Size 1000 -> 100)
  model.add_module('Hidden Linear2', nn.Linear(hidden2, hidden3))
  model.add_module('Hidden Activation2', nn.ReLU())

  # Hidden Layer (Size 100 -> 10)
  model.add_module('Hidden Linear3', nn.Linear(hidden3, output))
  model.add_module('Hidden Activation3', nn.ReLU())

  # Output Layer (Size 10 -> 1)
  model.add_module('Output Linear', nn.Linear(output, number_of_classes))
  model.add_module('Output Activation', nn.Sigmoid())

  # Loss Function
  loss_function = torch.nn.MSELoss()

  # Define a training procedure
  def train_network(train_data, actual_class, n):
      # Keep track of loss at every iteration
      loss_data = []
      converged = False #Set to n as default
      # Train for n iterations 
      for i in range(n):
          classification = model(train_data)

          # Calculate loss
          loss = loss_function(classification, actual_class)
          loss_data.append(loss)

          if loss.detach().numpy() < 0.025 and not converged:
              convergence_point = (i, loss.detach())
              converged = True
          # Zero out optimizer gradients every iteration
          optimizer.zero_grad()

          # Teach network how to increase performance in next iteration
          loss.backward()
          optimizer.step()
      if not converged:
        convergence_point = (n, loss.detach())

      return loss_data, convergence_point

  # Save networks default state to retrain from default weights
  torch.save(model, "model_default_state")

  ## Verify Network Works
  # Start from untrained every time
  model = torch.load("model_default_state")
  
  ## Retrieve Data from MNE EEG Dataset
  data_path = mne.datasets.sample.data_path()

  raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
  event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'

  p300s, others = extract_p300(raw_fname, event_fname)
  
  ## Prepare the train and test tensors
  # Specify Positive P300 train and test samples
  p300s_train = p300s[0:9]
  p300s_test = p300s[9:12]
  p300s_test = torch.tensor(p300s_test).float()
  p300s_test = p300s_test.detach().numpy()
  #p300s_test = p300s_test.detach().numpy()
  # Specify Negative P300 train and test samples
  others_train = others[30:39]
  others_test = others[39:42]
  others_test = torch.tensor(others_test).float()
  others_test = others_test.detach().numpy()
  #others_test = others_test.detach().numpy()
  # Combine everything into their final structures
  training_data = torch.tensor(np.concatenate((p300s_train, others_train), axis = 0)).float()
  positive_testing_data = torch.tensor(p300s_test).float()
  negative_testing_data = torch.tensor(others_test).float()

  # Print the size of each of our data structures
  # print("training data count: " + str(training_data.shape[0]))
  # print("positive testing data count: " + str(positive_testing_data.shape[0]))
  # print("negative testing data count: " + str(negative_testing_data.shape[0]))
  
  # Generate training labels
  labels = torch.tensor(np.zeros((training_data.shape[0],1))).detach().float()
  labels[0:10] = 1.0
  #print("training labels count: " + str(labels.shape[0]))


  ## Classify dataset w/ the Neural Network
  # Make sure we're starting from untrained every time
  model = torch.load("model_default_state")

  ## Define a learning function, needs to be reinitialized every load
  if optim == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
  elif optim == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=1)
  elif optim == "lbfgs":
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1, max_iter=20)
  else:
    raise Exception("Invalid optimizer")

  ## Use our training procedure with the sample data
  #print("Below is the loss graph for dataset training session")
  loss_data, convergence_point = train_network(training_data, labels, n = 50)

  # Classify our positive test dataset and print the results
  eeg_sample_length = 902
  classification_1 = model(positive_testing_data)
  
  correct = 0
  for index, value in enumerate(classification_1.data.tolist()):
      print("P300 Positive Classification {1}: {0:.2f}%".format(value[0] * 100, index + 1))
      if(value[0] > .5): correct+=1
  classification_2 = model(negative_testing_data)
  for index, value in enumerate(classification_2.data.tolist()):
      print("P300 Negative Classification {1}: {0:.2f}%".format(value[0] * 100, index + 1))
      if(value[0] < .5): correct+=1
  accuracy = correct/(len(others_test)+len(p300s_test))
  print(f"Accuracy: {100 * accuracy:.2f}%")
  
  return accuracy, loss_data, convergence_point
  # return positive_mean, negative_mean, accuracy, loss_data, convergence_point