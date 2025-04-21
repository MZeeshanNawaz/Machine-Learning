import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score

data = {
    'Day': ['sunny', 'windy', 'sunny', 'sunny', 'windy', 'sunny', 'sunny', 'windy', 'sunny', 'windy'],
    'Temp': ['hot', 'cold', 'hot', 'cold', 'hot', 'cold', 'hot', 'cold', 'hot', 'cold'],
    'class': ['play', 'Not play', 'play', 'Not play', 'play', 'play', 'play', 'Not play', 'play', 'Not play']
}
df = pd.DataFrame(data)
print(df)

X_raw = df[['Day', 'Temp']]
Y_raw = df['class']

onehot_encoder = OneHotEncoder()
x_encoded = onehot_encoder.fit_transform(X_raw).toarray()

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(Y_raw)

x_train, x_test, y_train, y_test = train_test_split(x_encoded, y_encoded, test_size=0.3, random_state=42)

# Model training with cross-validation
model1 = GaussianNB()
model1.fit(x_train, y_train)

# Hyperparameter tuning for Logistic Regression with reduced cv
param_grid = {'C': [0.1, 1, 10]}
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=2)
grid_search.fit(x_train, y_train)
best_model = grid_search.best_estimator_

# Evaluate models
y_pred = model1.predict(x_test)
print("Test Accuracy (GaussianNB):", accuracy_score(y_test, y_pred))

y_pred_best_model = best_model.predict(x_test)
print("Test Accuracy (Best Logistic Regression):", accuracy_score(y_test, y_pred_best_model))

print("Enter the day and temperature to predict the class:")
day = input("Enter the day: ")
temp = input("Enter the temperature: ")

# Validate user input
if day not in ['sunny', 'windy'] or temp not in ['hot', 'cold']:
    print("Invalid input. Please enter valid day and temperature.")
else:
    new_instance = pd.DataFrame([[day, temp]], columns=['Day', 'Temp'])
    new_instance_encoded = onehot_encoder.transform(new_instance).toarray()
    predicted_class = model1.predict(new_instance_encoded)
    predicted_label = label_encoder.inverse_transform(predicted_class)[0]
    print(f"Prediction for Day = {day}, Temp = {temp}: Result - {predicted_label}")
