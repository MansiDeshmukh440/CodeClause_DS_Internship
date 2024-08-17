import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

# Load the dataset
data = pd.read_csv("C:\\Users\\Acer\\Desktop\\CodeClause-2\\Heartdisease.csv")

# Feature and target variables
X = data.drop('target', axis=1)
y = data['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

import os
import matplotlib.pyplot as plt

# Other parts of your code for plotting and model evaluation

# Define the directory and file path
dir_path = 'static'
file_path = os.path.join(dir_path, 'confusion_matrix.png')

# Create the directory if it doesn't exist
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

# Plot and save the figure
plt.figure()
# ... (your plotting code here)
plt.savefig(file_path)  # This is where you save the figure

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('static/confusion_matrix.png')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        form_data = request.form

        # Convert form data to list of values
        input_data = [
            float(form_data['age']),
            float(form_data['sex']),
            float(form_data['cp']),
            float(form_data['trestbps']),
            float(form_data['chol']),
            float(form_data['fbs']),
            float(form_data['restecg']),
            float(form_data['thalach']),
            float(form_data['exang']),
            float(form_data['oldpeak']),
            float(form_data['slope']),
            float(form_data['ca']),
            float(form_data['thal'])
        ]

        # Reshape and scale input data
        input_data = np.array(input_data).reshape(1, -1)
        input_data = scaler.transform(input_data)

        # Predict using the model
        prediction = model.predict(input_data)

        # Convert prediction to human-readable format
        if prediction[0] == 1:
            prediction_text = "The patient is likely to have heart disease."
        else:
            prediction_text = "The patient is unlikely to have heart disease."

        # Render the result
        return render_template(
            'index.html', 
            prediction_text=prediction_text,
            accuracy=f"Model Accuracy: {accuracy*100:.2f}%"
        )
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)
