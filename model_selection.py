from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import pandas as pd

# Load and preprocess your dataset
#load data from a CSV into a pandas DataFrame
data = pd.read_csv('C://dev//opencv.test//Obesity_edited.csv')

#assuming the 'NObeyesdad' column contains the target variable, and the rest are features
X = data.drop('NObeyesdad', axis=1)  #features without 'NObeyesdad'
y = data['NObeyesdad']  #target variable 'NObeyesdad'

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

# Create a list of models to evaluate
models = [
    ("Random Forest", RandomForestClassifier(random_state=42)),
    ("Logistic Regression", LogisticRegression(random_state=42)),
    ("XGBoost", XGBClassifier(random_state=42)),
    ("SVM", SVC(random_state=42)),
    ("Neural Network", MLPClassifier(random_state=42))
]

# Initialize variables to store model results
best_model = None
best_accuracy = 0

# Model evaluation loop
for model_name, model in models:
    # Train the model on the training data
    model.fit(X_train, y_train)
    
    # Evaluate the model on the validation set
    y_pred_val = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred_val)
    
    # Record the model's performance
    print(f"{model_name} - Validation Accuracy: {accuracy:.4f}")
    
    # Check if this model is the best so far
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model_name

# Final evaluation of the best model on the test set
best_model = [model for model_name, model in models if model_name == best_model][0]
y_pred_test = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)
print(f"Best Model ({best_model}): Test Accuracy: {test_accuracy:.4f}")
