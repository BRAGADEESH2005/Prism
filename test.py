import os
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from script import extract_features  # Import extract_features from train.py
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the pre-trained model
model = joblib.load('trained_model.pkl')

# Define a function to preprocess the testing set
def preprocess_testing_set(dataset_directory, sr=16000, n_mfcc=13):
    feature_list = []

    # Path to the testing folder
    testing_path = os.path.join(dataset_directory, "testing")
    
    # Loop through 'real' and 'fake' subfolders
    for label in ["real", "fake"]:
        label_path = os.path.join(testing_path, label)
        for file in os.listdir(label_path):
            if file.endswith(".wav"):
                file_path = os.path.join(label_path, file)
                
                # Extract features
                features = extract_features(file_path, sr, n_mfcc)
                if features:
                    # Add label
                    features["label"] = label
                    feature_list.append(features)

    # Convert the features into a DataFrame
    df = pd.DataFrame(feature_list)
    return df

# Specify the dataset directory
dataset_directory = "./dataset"  # Replace with your dataset path

# Preprocess the testing set
test_df = preprocess_testing_set(dataset_directory)

# Split the features and labels
X_test = test_df.drop(columns=["label"])
y_test = test_df["label"]

# Predict using the trained model
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Print classification report
print(classification_report(y_test, y_pred))
