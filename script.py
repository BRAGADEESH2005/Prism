import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# Define a function to extract features from a single audio file
def extract_features(audio_path, sr=16000, n_mfcc=13):
    try:
        # Load the .wav file
        audio, _ = librosa.load(audio_path, sr=sr, mono=True)
        
        # Truncate or pad audio to 2 seconds
        target_length = sr * 2  # 2 seconds
        if len(audio) > target_length:
            audio = audio[:target_length]
        else:
            audio = np.pad(audio, (0, target_length - len(audio)))
        
        # Extract features
        features = {
            "zcr": np.mean(librosa.feature.zero_crossing_rate(audio)),
            "rmse": np.mean(librosa.feature.rms(y=audio)),  # Correct use of rms
            "energy": np.sum(audio**2),
            "spectral_centroid": np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)),
            "spectral_bandwidth": np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr)),
            "spectral_flatness": np.mean(librosa.feature.spectral_flatness(y=audio)),
            "spectral_rolloff": np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr)),
        }
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        for i in range(n_mfcc):
            features[f"mfcc_{i+1}"] = np.mean(mfccs[i])
        
        return features
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

# Define a function to preprocess the training set
def preprocess_training_set(dataset_directory, output_csv, sr=16000, n_mfcc=13):
    feature_list = []

    # Path to the training folder
    training_path = os.path.join(dataset_directory, "training")
    
    # Loop through 'real' and 'fake' subfolders
    for label in ["real", "fake"]:
        label_path = os.path.join(training_path, label)
        for file in tqdm(os.listdir(label_path), desc=f"Processing {label} files"):
            if file.endswith(".wav"):
                file_path = os.path.join(label_path, file)
                
                # Extract features
                features = extract_features(file_path, sr, n_mfcc)
                if features:
                    # Add label
                    features["label"] = label
                    feature_list.append(features)
    
    # Save features to a CSV file
    df = pd.DataFrame(feature_list)
    df.to_csv(output_csv, index=False)
    print(f"Features saved to {output_csv}")

# Specify the dataset directory and output CSV path
dataset_directory = "./dataset/"  # Replace with your dataset path
output_csv_path = "./training_features.csv"

# Preprocess the training set
preprocess_training_set(dataset_directory, output_csv_path)
