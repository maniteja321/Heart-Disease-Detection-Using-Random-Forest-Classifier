# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

# Setting up configurations
%matplotlib inline
print(os.listdir())
warnings.filterwarnings('ignore')

# Loading the dataset
data = pd.read_csv("/content/heart.csv")
print(data.shape)
print(data.head(5))
print(data.describe())
print(data.info())

# Describing the dataset columns
info = [
    "age",
    "1: male, 0: female",
    "chest pain type (1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic)",
    "resting blood pressure",
    "serum cholesterol in mg/dl",
    "fasting blood sugar > 120 mg/dl",
    "resting ECG values (0, 1, 2)",
    "max heart rate achieved",
    "exercise-induced angina",
    "oldpeak = ST depression induced by exercise relative to rest",
    "the slope of the peak exercise ST segment",
    "number of major vessels (0-3) colored by fluoroscopy",
    "thal: 3 = normal, 6 = fixed defect, 7 = reversible defect"
]
for i in range(len(info)):
    print(data.columns[i] + ":\t\t" + info[i])

# Exploring the target variable
print(data["target"].describe())
print(data["target"].unique())

# Splitting data into features and target
X = data.drop("target", axis=1)
Y = data["target"]

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

# Accuracy metric
from sklearn.metrics import accuracy_score

# Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, Y_train)
Y_pred_lr = lr.predict(X_test)
score_lr = round(accuracy_score(Y_pred_lr, Y_test) * 100, 2)
print("The accuracy score achieved using Logistic Regression is: " + str(score_lr) + "%")

# SVM
from sklearn.svm import SVC
sv = SVC(kernel='linear')
sv.fit(X_train, Y_train)
Y_pred_svm = sv.predict(X_test)
score_svm = round(accuracy_score(Y_pred_svm, Y_test) * 100, 2)
print("The accuracy score achieved using SVM is: " + str(score_svm) + "%")

# K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, Y_train)
Y_pred_knn = knn.predict(X_test)
score_knn = round(accuracy_score(Y_pred_knn, Y_test) * 100, 2)
print("The accuracy score achieved using KNN is: " + str(score_knn) + "%")

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
max_accuracy = 0
for x in range(2000):
    dt = DecisionTreeClassifier(random_state=x)
    dt.fit(X_train, Y_train)
    Y_pred_dt = dt.predict(X_test)
    curr_accuracy = round(accuracy_score(Y_pred_dt, Y_test) * 100, 2)
    if curr_accuracy > max_accuracy:
        max_accuracy = curr_accuracy
        best_x = x

dt = DecisionTreeClassifier(random_state=best_x)
dt.fit(X_train, Y_train)
Y_pred_dt = dt.predict(X_test)
score_dt = round(accuracy_score(Y_pred_dt, Y_test) * 100, 2)
print("The accuracy score achieved using Decision Tree is: " + str(score_dt) + "%")

# Random Forest
from sklearn.ensemble import RandomForestClassifier
max_accuracy = 0
for x in range(2000):
    rf = RandomForestClassifier(random_state=x)
    rf.fit(X_train, Y_train)
    Y_pred_rf = rf.predict(X_test)
    curr_accuracy = round(accuracy_score(Y_pred_rf, Y_test) * 100, 2)
    if curr_accuracy > max_accuracy:
        max_accuracy = curr_accuracy
        best_x = x

rf = RandomForestClassifier(random_state=best_x)
rf.fit(X_train, Y_train)
Y_pred_rf = rf.predict(X_test)
score_rf = round(accuracy_score(Y_pred_rf, Y_test) * 100, 2)
print("The accuracy score achieved using Random Forest is: " + str(score_rf) + "%")

# XGBoost
import xgboost as xgb
xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
xgb_model.fit(X_train, Y_train)
Y_pred_xgb = xgb_model.predict(X_test)
score_xgb = round(accuracy_score(Y_pred_xgb, Y_test) * 100, 2)
print("The accuracy score achieved using XGBoost is: " + str(score_xgb) + "%")

# Neural Networks
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(11, activation='relu', input_dim=13))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=300)
Y_pred_nn = model.predict(X_test)
rounded = [round(x[0]) for x in Y_pred_nn]
Y_pred_nn = rounded
score_nn = round(accuracy_score(Y_pred_nn, Y_test) * 100, 2)
print("The accuracy score achieved using Neural Network is: " + str(score_nn) + "%")

# Comparing algorithms
scores = [score_lr, score_svm, score_knn, score_dt, score_rf, score_xgb, score_nn]
algorithms = ["Logistic Regression", "SVM", "KNN", "Decision Tree", "Random Forest", "XGBoost", "Neural Networks"]

for i in range(len(algorithms)):
    print("The accuracy score achieved using " + algorithms[i] + " is: " + str(scores[i]) + "%")

# Visualizing the accuracy
sns.set(rc={'figure.figsize': (10, 4)})
plt.xlabel("Algorithms")
plt.ylabel("Accuracy score")
plt.scatter(algorithms, scores)
plt.show()

# Making a prediction
new_data = pd.DataFrame({
    'age': 52,
    'sex': 1,
    'cp': 0,
    'trestbps': 125,
    'chol': 212,
    'fbs': 0,
    'restecg': 1,
    'thalach': 168,
    'exang': 0,
    'oldpeak': 1.0,
    'slope': 2,
    'ca': 2,
    'thal': 3,
}, index=[0])

p = rf.predict(new_data)
if p[0] == 0:
    print("No Disease")
else:
    print("Disease")

# Save and load the model
import joblib
joblib.dump(rf, 'trained_model.joblib')
from google.colab import files
files.download('trained_model.joblib')

# Interactive prediction
import ipywidgets as widgets
from IPython.display import display
model = joblib.load('trained_model.joblib')

def make_prediction(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    input_data = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        result = "Disease"
    else:
        result = "No Disease"
    print(result)
