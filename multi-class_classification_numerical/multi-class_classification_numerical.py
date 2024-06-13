from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import pandas as pd

data = pd.read_csv('C://dev//opencv.test//Obesity_edited.csv')

#assuming the 'NObeyesdad' column contains the target variable, and the rest are features
X = data.drop('NObeyesdad', axis=1)  #features without 'NObeyesdad'
y = data['NObeyesdad']  #target variable 'NObeyesdad'

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

models = [
    ("Random Forest", RandomForestClassifier(random_state=42)),
    ("Logistic Regression", LogisticRegression(random_state=42)),
    ("XGBoost", XGBClassifier(random_state=42)),
    ("SVM", SVC(random_state=42)),
    ("Neural Network", MLPClassifier(random_state=42))
]

best_model = None
best_accuracy = 0

for model_name, model in models:
    model.fit(X_train, y_train)
    y_pred_val = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred_val)
    print(f"{model_name} - Validation Accuracy: {accuracy:.4f}")
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model_name

best_model = [model for model_name, model in models if model_name == best_model][0]
y_pred_test = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)
print(f"Best Model ({best_model}): Test Accuracy: {test_accuracy:.4f}")
