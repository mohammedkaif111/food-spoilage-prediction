from flask import Flask, request, render_template
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import pickle

# Initialize the Flask app
app = Flask(__name__)

# Load the trained models and the scaler (assumed to be saved as pickle files)
with open('classifier_model.pkl', 'rb') as f:
    clf = pickle.load(f)

with open('regressor_model.pkl', 'rb') as f:
    reg = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Define a function for prediction
def predict_food_spoilage(features):
    # Scale the input features
    features_scaled = scaler.transform([features])
    
    # Classification first (Is the food spoiled?)
    is_spoiled = clf.predict(features_scaled)[0]
    
    if is_spoiled:
        return "Food is spoiled."
    else:
        # If not spoiled, predict the time left before spoilage
        time_left = reg.predict(features_scaled)[0]
        return f"Food is not spoiled. Estimated time to spoil: {time_left:.2f} days."

# Define the routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form inputs (temperature, moisture, gasvalue)
        temperature = float(request.form['temperature'])
        moisture = float(request.form['moisture'])
        gasvalue = float(request.form['gasvalue'])
        
        # Prepare features
        features = [temperature, moisture, gasvalue]
        
        # Make the prediction
        prediction = predict_food_spoilage(features)
        
        return render_template('index.html', prediction_text=prediction)
    
    except ValueError:
        return render_template('index.html', prediction_text="Invalid input. Please enter valid numbers.")

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
