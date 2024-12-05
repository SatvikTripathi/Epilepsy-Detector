import os
import pickle
import numpy as np
import pandas as pd
import mne
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

app = Flask(__name__)   # Initializing flask app
app.secret_key = 'epilepsy'

# handling uploads
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# loading the model

model_path = 'models/xgboost_model.pkl'
with open(model_path, 'rb') as file:
    xgb_model = pickle.load(file)


# checking uploaded file's extension
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# functions for working with EEG data

# converting dataframe to mne object
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
    """EEG relative power band feature extraction."""
    FREQ_BANDS = {
        "delta": [0.5, 4.5],
        "theta": [4.5, 8.5],
        "alpha": [8.5, 11.5],
        "sigma": [11.5, 15.5],
        "beta": [15.5, 30],
        "gamma": [30, 45],
    }

    # Compute the PSD using Welch's method
    psd = epochs.compute_psd(method="welch", fmin=0.5, fmax=45, picks="eeg")
    psds, freqs = psd.get_data(return_freqs=True)

    # Normalize the PSDs
    psds /= np.sum(psds, axis=-1, keepdims=True)

    # Extract mean PSD for each frequency band
    X = []
    for fmin, fmax in FREQ_BANDS.values():
        freq_mask = (freqs >= fmin) & (freqs < fmax)
        psds_band = psds[:, :, freq_mask].mean(axis=-1)
        X.append(psds_band)

    return np.concatenate(X, axis=1)

# defining methods for input output
@app.route('/', methods=['GET', 'POST'])

def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']

        # If user does not select file, browser may submit an empty part
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Get additional parameters
            record_time = request.form.get('record_time')  # e.g., duration in seconds
            eye_on = request.form.get('eye_on')  # e.g., seconds eye is on
            eye_off = request.form.get('eye_off')  # e.g., seconds eye is off

            try:
                # Convert record_time and eye condition parameters to integers
                record_time = int(record_time)
                eye_on = int(eye_on)
                eye_off = int(eye_off)
            except ValueError:
                flash('Record time and eye condition parameters must be integers.')
                return redirect(request.url)

            # Process the EEG data and make prediction
            try:
                prediction = process_and_predict(filepath, record_time, eye_on, eye_off)
                # Remove the uploaded file after processing
                os.remove(filepath)
                return render_template('result.html', prediction=prediction)
            except Exception as e:
                flash(f'An error occurred during processing: {e}')
                return redirect(request.url)

    return render_template('index.html')

def process_and_predict(filepath, record_time, eye_on, eye_off):
    """Process the EEG CSV file and make a prediction using the trained model."""
    # Read the CSV file
    df = pd.read_csv(filepath)

    # Depending on how 'record_time' and 'eye_condition' affect your data,
    # you may need to preprocess the data accordingly.
    # For example, you might segment the data based on eye conditions.
    # This part needs to be tailored to your specific application.

    # Example: If 'record_time' is the total duration, ensure the data matches
    expected_samples = record_time * 128  # Assuming 128 Hz sampling rate
    if df.shape[0] < expected_samples:
        raise ValueError('Insufficient data for the specified record time.')
    df = df.iloc[:expected_samples]

    # If 'eye_on' and 'eye_off' are used to segment data, implement accordingly
    # For simplicity, we'll proceed without altering based on eye conditions
    # Modify this part based on your actual requirements

    # Remove non-EEG channels if necessary (assuming channels are in specific columns)
    df = df.iloc[:, 1:15]  # Adjust if your CSV structure is different

    # Convert to MNE Epochs
    epochs = convertDF2MNE(df, sfreq=128, duration=5, overlap=1)

    if len(epochs) == 0:
        raise ValueError('No valid epochs found after processing.')

    # Extract features
    features = eeg_power_band(epochs)

    # Make predictions
    predictions = xgb_model.predict(features)

    # Aggregate predictions (e.g., majority vote)
    final_prediction = np.round(np.mean(predictions)).astype(int)

    # Map label to meaningful output
    label_map = {0: 'Epilepsy', 1: 'Control'}
    return label_map.get(final_prediction, 'Unknown')

if __name__ == '__main__':
    app.run(debug=True)