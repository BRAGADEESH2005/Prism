import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the training dataset
df = pd.read_csv("./training_features.csv")  # Path to your CSV file

# Split the data into features (X) and labels (y)
X = df.drop(columns=["label"])
y = df["label"]

# Initialize the RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the entire dataset
model.fit(X, y)

# Save the trained model to a .pkl file
with open("trained_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved as 'trained_model.pkl'")
