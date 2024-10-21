# Importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error

# Load the updated dataset
df = pd.read_csv(r'synthetic_food_spoilage_india.csv')

# Step 1: Split the data into features and targets
X = df[['temperature', 'moisture', 'gas']]  # Input features
y_classification = df['spoiled']  # Target for classification (spoiled or not)
y_regression = df['time_to_spoil']  # Target for regression (time to spoil)

# Step 2: Train-test split for classification (detect if spoiled or not)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X, y_classification, test_size=0.2, random_state=42)

# Step 3: Train-test split for regression (only for non-spoiled entries)
non_spoiled_df = df[df['spoiled'] == 0]
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(non_spoiled_df[['temperature', 'moisture', 'gas']], 
                                                                    non_spoiled_df['time_to_spoil'], 
                                                                    test_size=0.2, random_state=42)

# Step 4: Standardize the features (useful for most ML models)
scaler = StandardScaler()
X_train_clf = scaler.fit_transform(X_train_clf)
X_test_clf = scaler.transform(X_test_clf)
X_train_reg = scaler.fit_transform(X_train_reg)
X_test_reg = scaler.transform(X_test_reg)

# Step 5: Build and train a Random Forest Classifier (for spoiled/not spoiled prediction)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_clf, y_train_clf)

# Step 6: Evaluate the classifier
y_pred_clf = clf.predict(X_test_clf)
accuracy = accuracy_score(y_test_clf, y_pred_clf)
print(f"Classification Accuracy (Spoiled/Not Spoiled): {accuracy * 100:.2f}%")

# Step 7: Build and train a Random Forest Regressor (for time to spoil prediction)
reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train_reg, y_train_reg)

# Step 8: Evaluate the regressor
y_pred_reg = reg.predict(X_test_reg)
mse = mean_squared_error(y_test_reg, y_pred_reg)
print(f"Regression Mean Squared Error (Time to Spoilage): {mse:.2f}")

# Step 9: Combined prediction function
def predict_food_spoilage(features):
    # features: [temperature, moisture, gasvalue]
    features_scaled = scaler.transform([features])
    
    # Classification first (Is the food spoiled?)
    is_spoiled = clf.predict(features_scaled)[0]
    
    if is_spoiled:
        return "Food is spoiled."
    else:
        # If not spoiled, predict the time left before spoilage
        time_left = reg.predict(features_scaled)[0]
        return f"Food is not spoiled. Estimated time to spoil: {time_left:.2f} days."

# Example prediction (replace with actual inputs)
example_features = [25, 65, 300]  # Example: temperature=25°C, moisture=65%, gasvalue=300 ppm
result = predict_food_spoilage(example_features)
print(result)
import pickle

# Assuming clf (classifier), reg (regressor), and scaler (scaler) are already trained

# Save the models
with open('classifier_model.pkl', 'wb') as f:
    pickle.dump(clf, f)

with open('regressor_model.pkl', 'wb') as f:
    pickle.dump(reg, f)

# Save the scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
