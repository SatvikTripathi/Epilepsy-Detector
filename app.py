import os
import pickle
import numpy as np
import pandas as pd
import mne
import streamlit as st

# Set up Streamlit app
st.set_page_config(page_title="EEG Analysis", layout="centered")
st.title("EEG Analysis for Epilepsy Detection")

# Load the model
model_path = 'models/xgboost_model.pkl'
with open(model_path, 'rb') as file:
    xgb_model = pickle.load(file)

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv'}

def convertDF2MNE(sub, sfreq=128, duration=5, overlap=1):
    info = mne.create_info(list(sub.columns), ch_types=['eeg'] * len(sub.columns), sfreq=sfreq)
    info.set_montage('standard_1020')
    data = mne.io.RawArray(sub.T, info)
    data.set_eeg_reference()
    data.filter(l_freq=0.1, h_freq=45)
    epochs = mne.make_fixed_length_epochs(data, duration=duration, overlap=overlap)
    epochs = epochs.drop_bad()
    return epochs

def eeg_power_band(epochs):
    FREQ_BANDS = {
        "delta": [0.5, 4.5],
        "theta": [4.5, 8.5],
        "alpha": [8.5, 11.5],
        "sigma": [11.5, 15.5],
        "beta": [15.5, 30],
        "gamma": [30, 45],
    }
    psd = epochs.compute_psd(method="welch", fmin=0.5, fmax=45, picks="eeg")
    psds, freqs = psd.get_data(return_freqs=True)
    psds /= np.sum(psds, axis=-1, keepdims=True)
    X = []
    for fmin, fmax in FREQ_BANDS.values():
        freq_mask = (freqs >= fmin) & (freqs < fmax)
        psds_band = psds[:, :, freq_mask].mean(axis=-1)
        X.append(psds_band)
    return np.concatenate(X, axis=1)

def process_and_predict(filepath, record_time, eye_on, eye_off):
    df = pd.read_csv(filepath)
    expected_samples = record_time * 128
    if df.shape[0] < expected_samples:
        raise ValueError('Insufficient data for the specified record time.')
    df = df.iloc[:expected_samples]
    df = df.iloc[:, 1:15]
    epochs = convertDF2MNE(df, sfreq=128, duration=5, overlap=1)
    if len(epochs) == 0:
        raise ValueError('No valid epochs found after processing.')
    features = eeg_power_band(epochs)
    predictions = xgb_model.predict(features)
    final_prediction = np.round(np.mean(predictions)).astype(int)
    label_map = {0: 'Epilepsy', 1: 'Control'}
    return label_map.get(final_prediction, 'Unknown')

# Streamlit UI
uploaded_file = st.file_uploader("Upload EEG Data (CSV)", type="csv")
record_time = st.number_input("Record Time (seconds)", min_value=1, step=1)
eye_on = st.number_input("Eye On Time (seconds)", min_value=0, step=1)
eye_off = st.number_input("Eye Off Time (seconds)", min_value=0, step=1)

# Function to plot PSD
def plot_psd(raw_data, channel_names, sfreq=128):
    """
    Plot the Power Spectral Density (PSD) of EEG data.

    Parameters:
    - raw_data: numpy.ndarray of shape (n_channels, n_samples), EEG data.
    - channel_names: list of str, names of EEG channels.
    - sfreq: float, sampling frequency of the EEG data.
    """
    info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(raw_data, info)
    raw.set_montage('standard_1020')

    # Plot PSD
    fig = raw.plot_psd(fmin=0.1, fmax=45, show=False)  # Customize frequency range as needed
    return fig

# Streamlit prediction logic
if st.button("Predict"):
    if uploaded_file is not None:
        try:
            filepath = os.path.join("uploads", uploaded_file.name)
            with open(filepath, "wb") as f:
                f.write(uploaded_file.getbuffer())

            prediction = process_and_predict(filepath, record_time, eye_on, eye_off)

            # Display prediction
            st.success(f"Prediction: {prediction}")

            # Create MNE raw object for PSD
            df = pd.read_csv(filepath).iloc[:, 1:15]  # Select only EEG data columns
            raw_data = df.to_numpy().T  # Transpose to shape (n_channels, n_samples)
            channel_names = list(df.columns)

            # Plot and display PSD
            st.subheader("Power Spectral Density (PSD)")
            fig_psd = plot_psd(raw_data, channel_names, sfreq=128)
            st.pyplot(fig_psd)

            # Remove uploaded file
            os.remove(filepath)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.error("Please upload a CSV file.")
