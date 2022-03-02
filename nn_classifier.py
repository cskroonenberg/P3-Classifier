from collections import OrderedDict
from pylab import rcParams
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import mne
from sklearn.preprocessing import RobustScaler

# See http://learn.neurotechedu.com/machinelearning/

# Set randomizer seed for consistency
torch.manual_seed(100)

eeg_sample_count = 240 # Number of samples to train network with
learning_rate = 1e-3 # How fast the network will change its weights
eeg_sample_length = 226 # Number of datapoints per sample
number_of_classes = 1 # 1 Output variable w/ 1.0 = 100%, 0.0 = 0% certainty that sample has p300
hidden1 = 500 # Number of neurons in first hidden layer
hidden2 = 1000 # Number of neurons in second hidden layer
hidden3 = 100 # Number of neurons in third hidden layer
output = 10 # Number of neurons in output layer

## Create sample data
sample_positives = [None, None] # Element [0] is the sample, Element [1] is the class
sample_positives[0] = torch.rand(int(eeg_sample_count / 2), eeg_sample_length) * 0.50 + 0.25
sample_positives[1] = torch.ones([int(eeg_sample_count / 2), 1], dtype=torch.float32)

sample_negatives = [None, None] # Element [0] is the sample, Element [1] is the class
sample_negatives_low = torch.rand(int(eeg_sample_count / 4), eeg_sample_length) * 0.25
sample_negatives_high = torch.rand(int(eeg_sample_count / 4), eeg_sample_length) * 0.25 + 0.75
sample_negatives[0] = torch.cat([sample_negatives_low, sample_negatives_high], dim = 0)
sample_negatives[1] = torch.zeros([int(eeg_sample_count / 2), 1], dtype=torch.float32)

samples = [None, None] # Combine the two
samples[0] = torch.cat([sample_positives[0], sample_negatives[0]], dim = 0)
samples[1] = torch.cat([sample_positives[1], sample_negatives[1]], dim = 0)

## Create test data that isn't trained on
test_positives = torch.rand(10, eeg_sample_length) * 0.50 + 0.25 # Test 10 good samples
test_negatives_low = torch.rand(5, eeg_sample_length) * 0.25 # Test 5 bad low samples
test_negatives_high = torch.rand(5, eeg_sample_length) * 0.25 + 0.75 # Test 5 bad high samples
test_negatives = torch.cat([test_negatives_low, test_negatives_high], dim = 0)

print("We have created a sample dataset with " + str(samples[0].shape[0]) + " samples")
print("Half of those are positive samples with a score of 100%")
print("Half of those are negative samples with a score of 0%")
print("We have also created two sets of 10 test samples to check the validity of the network")

## Network
model = nn.Sequential()

# Input layer (Size 226 -> 500)
model.add_module('Input Linear', nn.Linear(eeg_sample_length, hidden1))
model.add_module('Input Activation', nn.CELU())

# Hidden Layer (Size 500 -> 1000)
model.add_module('Hidden Linear', nn.Linear(hidden1, hidden2))
model.add_module('Hidden Activation', nn.ReLU())

# Hidden Layer (Size 1000 -> 100)
model.add_module('Hidden Linear2', nn.Linear(hidden2, hidden3))
model.add_module('Hidden Activation2', nn.ReLU())

# Hidden Layer (Size 100 -> 10)
model.add_module('Hidden Linear3', nn.Linear(hidden3, 10))
model.add_module('Hidden Activation3', nn.ReLU())

# Output Layer (Size 10 -> 1)
model.add_module('Output Linear', nn.Linear(10, number_of_classes))
model.add_module('Output Activation', nn.Sigmoid())

# Loss Function
loss_function = torch.nn.MSELoss()

# Define a training procedure
def train_network(train_data, actual_class, n):
    # Keep track of loss at every iteration
    loss_data = []

    # Train for n iterations 
    for i in range(n):
        classification = model(train_data)

        # Calculate loss
        loss = loss_function(classification, actual_class)
        loss_data.append(loss)

        # Zero out optimizer gradients every iteration
        optimizer.zero_grad()

        # Teach network how to increase performance in next iteration
        loss.backward()
        optimizer.step()

    # Plot loss graph at end of training
    # rcParams['figure.figsize'] = 10, 5
    # plt.title("Loss Vs Iterations")
    # plt.plot(list(range(0, len(loss_data))), loss_data)
    # plt.show()

# Save networks default state to retrain from default weights
torch.save(model, "model_default_state")

## Verify Network Works
# Start from untrained every time
model = torch.load("model_default_state")

# Define a learning function, needs to be reinitialized every load
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

# Train the network using our training procedure with the sample data
train_network(samples[0], samples[1], n = 100)

# Classify our positive test dataset
predicted_positives = model(test_positives).data.tolist()

# Print the results
for index, value in enumerate(predicted_positives):
    print("Positive Test {1} Value scored: {0:.2f}%".format(value[0] * 100, index + 1))

print()

#Classify the negative test dataset
predicted_negatives = model(test_negatives).data.tolist()

# Print the results
for index, value in enumerate(predicted_negatives):
    print("Negative Test {1} Value scored: {0:.2f}%".format(value[0] * 100, index + 1))

rcParams['figure.figsize'] = 10, 5
plt.scatter(list(range(0, eeg_sample_length)), test_positives[3], color = "#00aa00")
plt.plot(list(range(0, eeg_sample_length)), test_positives[3], color = "#bbbbbb")
plt.scatter(list(range(0, eeg_sample_length)), test_negatives[0], color = "#aa0000")
plt.plot(list(range(0, eeg_sample_length)), test_negatives[0], color = "#bbbbbb")
plt.scatter(list(range(0, eeg_sample_length)), test_negatives[9], color = "#aa0000")
plt.plot(list(range(0, eeg_sample_length)), test_negatives[9], color = "#bbbbbb")
plt.ylim([0 , 1])
# plt.show()

## Retrieve Data from MNE EEG Dataset
data_path = mne.datasets.sample.data_path()

raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'

# Obtain a reference to the database and preload into RAM
raw_data = mne.io.read_raw_fif(raw_fname, preload=True) 

# EEGs work by detecting the voltage between two points. The second reference
# point is set to be the average of all voltages using the following function.
# It is also possible to set the reference voltage to a different number.
raw_data.set_eeg_reference()

# Define what data we want from the dataset
raw_data = raw_data.pick(picks=["eeg","eog"])
picks_eeg_only = mne.pick_types(raw_data.info, eeg=True, eog=True, meg=False, exclude='bads')

# Gather events
events = mne.read_events(event_fname)
event_id = 5
tmin = -0.5 
tmax = 1
epochs = mne.Epochs(raw_data, events, event_id, tmin, tmax, proj=True,
                    picks=picks_eeg_only, baseline=(None, 0), preload=True,
                    reject=dict(eeg=100e-6, eog=150e-6), verbose = False)

# This is the channel used to monitor the P300 response
channel = "EEG 058"

# Display a graph of the sensor position we're using
sensor_position_figure = epochs.plot_sensors(show_names=[channel])

event_id=[1,2,3,4]
epochsNoP300 = mne.Epochs(raw_data, events, event_id, tmin, tmax, proj=True,
                    picks=picks_eeg_only, baseline=(None, 0), preload=True,
                    reject=dict(eeg=100e-6, eog=150e-6), verbose = False)

# mne.viz.plot_compare_evokeds({'P300': epochs.average(picks=channel), 'Other': epochsNoP300[0:12].average(picks=channel)})

eeg_data_scaler = RobustScaler()

# We have 12 p300 samples
p300s = np.squeeze(epochs.get_data(picks=channel))

# We have 208 non-p300 samples
others = np.squeeze(epochsNoP300.get_data(picks=channel))

# Scale the p300 data using the RobustScaler
p300s = p300s.transpose()
p300s = eeg_data_scaler.fit_transform(p300s)
p300s = p300s.transpose()

# Scale the non-p300 data using the RobustScaler
others = others.transpose()
others = eeg_data_scaler.fit_transform(others)
others = others.transpose()

## Prepare the train and test tensors
# Specify Positive P300 train and test samples
p300s_train = p300s[0:9]
p300s_test = p300s[9:12]
p300s_test = torch.tensor(p300s_test).float()

# Specify Negative P300 train and test samples
others_train = others[30:39]
others_test = others[39:42]
others_test = torch.tensor(others_test).float()

# Combine everything into their final structures
training_data = torch.tensor(np.concatenate((p300s_train, others_train), axis = 0)).float()
positive_testing_data = torch.tensor(p300s_test).float()
negative_testing_data = torch.tensor(others_test).float()

# Print the size of each of our data structures
print("training data count: " + str(training_data.shape[0]))
print("positive testing data count: " + str(positive_testing_data.shape[0]))
print("negative testing data count: " + str(negative_testing_data.shape[0]))

# Generate training labels
labels = torch.tensor(np.zeros((training_data.shape[0],1))).float()
labels[0:10] = 1.0
print("training labels count: " + str(labels.shape[0]))


## Classify dataset w/ the Neural Network
# Make sure we're starting from untrained every time
model = torch.load("model_default_state")

## Define a learning function, needs to be reinitialized every load
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

## Use our training procedure with the sample data
print("Below is the loss graph for dataset training session")
train_network(training_data, labels, n = 50)

# Classify our positive test dataset and print the results
classification_1 = model(positive_testing_data)
for index, value in enumerate(classification_1.data.tolist()):
    print("P300 Positive Classification {1}: {0:.2f}%".format(value[0] * 100, index + 1))

print()

# Classify our negative test dataset and print the results
classification_2 = model(negative_testing_data)
for index, value in enumerate(classification_2.data.tolist()):
    print("P300 Negative Classification {1}: {0:.2f}%".format(value[0] * 100, index + 1))