from sklearn.preprocessing import RobustScaler
import mne
import numpy as np

def extract_p300(raw_fname, event_fname):
    """
    """
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
    #sensor_position_figure = epochs.plot_sensors(show_names=[channel])

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

    return p300s, others
