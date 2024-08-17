import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
file_path = "C:\\Users\\Acer\\Desktop\\codeClause\\parkinsons.csv"
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(data.head())

# Features and Target variable
X = data.drop(['name', 'status'], axis=1)  # Assuming 'status' is the target variable
y = data['status']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training using RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model Prediction
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the model and scaler for use in the Flask app
import joblib
joblib.dump(model, 'parkinsons_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
