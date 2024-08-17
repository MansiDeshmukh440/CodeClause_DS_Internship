from flask import Flask, render_template, request
import joblib
import numpy as np

# Load the model and scaler
model = joblib.load('parkinsons_model.pkl')
scaler = joblib.load('scaler.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Log received form values
        print("Form data received:", request.form)

        # Extracting the input features from the form
        features = [float(x) for x in request.form.values()]
        print("Processed features:", features)

        final_features = [np.array(features)]
        scaled_features = scaler.transform(final_features)
        print("Scaled features:", scaled_features)

        # Predicting using the loaded model
        prediction = model.predict(scaled_features)
        print("Prediction result:", prediction)

        # Interpreting the result
        result = 'Parkinson\'s Disease Detected' if prediction[0] == 1 else 'No Parkinson\'s Disease Detected'
        return render_template('index.html', prediction_text=result)
    
    except Exception as e:
        # Log any errors that occur
        print(f"Error occurred: {e}")
        return render_template('index.html', prediction_text="An error occurred. Please try again.")

if __name__ == "__main__":
    app.run(debug=True)
