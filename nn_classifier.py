import torch
import torch.nn as nn
import numpy as np
import mne
import time
from extract_positives import extract_p300
from sklearn.model_selection import train_test_split
from imblearn import over_sampling

# See http://learn.neurotechedu.com/machinelearning/

def train_and_test(learning_rate, hidden1, hidden2, hidden3, output, extra_layers=0, optim="adam", n=250):
  """
  Function to train and test the neural network with customized parameters.
  :param learning_rate: Customized learning rate for training (backprop) as a float.
  :param hidden1: Size of 1st hidden layer as an int
  :param hidden2: Size of 2nd hidden layer as an int
  :param hidden3: Size of 3rd hidden layer as an int
  :param extra_layers: Number of additional layers to add to the nn
  :param optim: Optimization function to chose as a string
  :return: Averages of testing scores, convergence times, but matplot lib plots testing results.
  """
  # Set randomizer seed for consistency
  torch.manual_seed(100)
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
  def train_network(train_data, labels, n):
      # Keep track of loss at every iteration
    
      loss_data = []
      converged = False #Set to n as default
      # Train for n iterations 
      for i in range(n):
          classification = model(train_data)
          
          # Calculate loss
          loss = loss_function(classification, labels) # Error here
          loss_data.append(loss.detach().numpy())
          if loss.detach().numpy() < 0.005 and not converged:
              convergence_point = (i, loss.detach())
              converged = True
              break
          # Zero out optimizer gradients every iteration
          optimizer.zero_grad()

          # Teach network how to increase performance in next iteration
          loss.backward()
          optimizer.step()
      if not converged:
        convergence_point = (n, loss.detach())

      return loss_data, convergence_point
  
  def train_lbfgs(optimizer, train_data, labels, n):
    loss_data = []
    n = 25 #reset n
    for i in range(n):
      def closure():
        if torch.is_grad_enabled():
          optimizer.zero_grad()
        classification = model(train_data)
        loss = loss_function(classification, labels)
        loss_data.append(loss.detach().numpy())
        if loss.requires_grad:
          loss.backward()
        return loss
      optimizer.step(closure)
  
    convergence_point = (22, 0.025) #Dummy variable - #NOTE only used for original size

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
  
  # combine datasets
  p300s, others = extract_p300(raw_fname, event_fname)
  X = np.concatenate((p300s, others))

  # combine labels
  p300_labels = [0 for i in range(len(p300s))]
  others_labels = [1 for i in range(len(others))]
  y = np.concatenate((p300_labels, others_labels))

  # oversampling
  X, y = over_sampling.ADASYN().fit_resample(X, y)

  # split data into training and test datasets
  X_trainFold1, X_testFold1, y_trainFold1, y_testFold1 = train_test_split(X, y, test_size=0.50, random_state=1)

  # Convert data to tensors
  X_trainTensor1 = torch.from_numpy(X_trainFold1).float()
  X_testTensor1 = torch.from_numpy(X_testFold1).float()
  y_trainTensor1 = torch.FloatTensor(y_trainFold1).float()
  y_testTensor1 = torch.FloatTensor(y_testFold1).float()

  # transform y tensors to have proper size
  new_shape = (len(y_trainFold1), 1)
  y_trainTensor1 = y_trainTensor1.view(new_shape)
  y_testTensor1 = y_testTensor1.view(new_shape)

  ## Classify dataset w/ the Neural Network
  model = torch.load("model_default_state") # Make sure we're starting from untrained every time

  ## Define a learning function, needs to be reinitialized every load
  t0 = time.time()
  if optim == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
  elif optim == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=1)
  elif optim == "lbfgs":
    optimizer = torch.optim.LBFGS(model.parameters(), line_search_fn='strong_wolfe')
    loss_data, convergence_point = train_lbfgs(optimizer, X_trainTensor1, y_trainTensor1, n)
  elif optim == "asgd":
    optimizer = torch.optim.ASGD(model.parameters())
  elif optim == "adamax":
    optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)
  elif optim == "rprop":
    optimizer = torch.optim.Rprop(model.parameters(), lr=learning_rate)
  else:
    raise Exception("Invalid optimizer")

  ## Train NN
  if optim != "lbfgs":
    loss_data, convergence_point = train_network(X_trainTensor1, y_trainTensor1, n)
  t1 = time.time()
  convergence_time = t1-t0
  # Predict labels for test dataset
  y_pred = model(X_testTensor1)

  # compare prediction to actual
  TP = 0
  TN = 0
  FN = 0
  FP = 0
  for i, value in enumerate(y_pred.data.tolist()):
      if value[0] <= .5:
        if y_testTensor1[i] == 0: TN+=1
        else: FN+=1
      if value[0] > .5:
        if y_testTensor1[i] == 1: TP+=1
        else: FP+=1

  # calculate accuracy
  accuracy = (TP + TN)/(TP + TN + FP + FN)
  precision = TP/(TP + FP)
  recall = TP/(TP + FN)
  f1_score = 2 * ((precision * recall) / (precision + recall))
  print("\n-----------------\n  Scores:\n-----------------")
  print(f"Accuracy: {100 * accuracy:.2f}%")
  print(f"Precision: {100 * precision:.2f}%")
  print(f"Recall: {100 * recall:.2f}%")
  print(f"F1 Score: {100 * f1_score:.2f}%")
  print("-----------------")
  return accuracy, loss_data, convergence_point, convergence_time