import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import  RidgeClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score  
from sklearn.model_selection import cross_val_score             
from sklearn.model_selection import GridSearchCV


data = {
    'Day':['sunny','windy','sunny','sunny','windy','sunny','sunny','windy','sunny','windy'],
    'Temp':['hot','cold','hot','cold','hot','cold','hot','cold','hot','cold'],
    'class':['play','Not play','play','Not play','play','play','play','Not play','play','Not play']
}
df = pd.DataFrame(data)
print(df)
X_raw = df[['Day','Temp']]
Y_raw = df['class']

Onehot_encoder = OneHotEncoder()
x_encoded = Onehot_encoder.fit_transform(X_raw).toarray()

label_Encoder = LabelEncoder()
y_encoded = label_Encoder.fit_transform(Y_raw)

x_train,x_test, y_train ,y_test = train_test_split(x_encoded,y_encoded,test_size=0.3,random_state=42)
model1 = GaussianNB()
model1.fit(x_train,y_train)

model2 = LogisticRegression()
model2.fit(x_train,y_train)

model3 = RidgeClassifier()
model3.fit(x_train,y_train)

model4 = SVC()
model4.fit(x_train,y_train)



y_pred = model1.predict(x_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))

print("Enter the day and temperature to predict the class:")
day = input("Enter the day: ")
temp = input("Enter the temperature: ")

new_instance = pd.DataFrame([[day,temp]], columns=['Day','Temp'])
new_instance_encoded = Onehot_encoder.transform(new_instance).toarray()
predicted_class = model1.predict(new_instance_encoded)
predicted_label = label_Encoder.inverse_transform(predicted_class)[0]
print("Prediction for Day = Windy,temp = cool: Result",predicted_label)

