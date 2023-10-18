# import libraries
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# load the Breast Cancer dataset
cancer = load_breast_cancer()

# define features as X and target as y
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = cancer.target

# split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# normalize data to be between 0 and 1
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# list of models to evaluate
models = [
    ("Random Forest", RandomForestClassifier(random_state=42)),
    ("Logistic Regression", LogisticRegression(random_state=42)),
    ("XGBoost", XGBClassifier(random_state=42)),
    ("SVM", SVC(random_state=42)),
    ("Neural Network", MLPClassifier(random_state=42)),
    ("Decision Tree", DecisionTreeClassifier(random_state=42)),
]

# initialize variables to store the best model and accuracy
best_model = None
best_accuracy = 0

# model evaluation loop
for model_name, model in models:
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} - Validation Accuracy: {accuracy:.4f}")
    
    # check if this model has the best accuracy so far
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model_name

# final evaluation of the best model on the test set
best_classifier = [model for model_name, model in models if model_name == best_model][0]
y_pred_test = best_classifier.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_pred_test)
print(f"Best Model ({best_model}): Test Accuracy: {test_accuracy:.4f}")
