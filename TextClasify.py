import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

data = {
    'text': [
        'This is a sample text',
        'I am a good boy',
        'Win a free iPhone now!',
        'Click here to claim your prize',
        'Your appointment is confirmed',
        'Limited time offer just for you',
        'Reminder: meeting at 3PM',
        'Congratulations, you have won!',
        'Let’s catch up tomorrow',
        'Earn money fast from home'
    ],
    'label': [
        'ham',
        'ham',
        'spam',
        'spam',
        'ham',
        'spam',
        'ham',
        'spam',
        'ham',
        'spam'
    ]
}

df = pd.DataFrame(data)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df['label'])

vectorizer = TfidfVectorizer()
x_encoded = vectorizer.fit_transform(df['text'])

x_train, x_test, y_train, y_test = train_test_split(
    x_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Actual label distribution:", np.bincount(y_test))
print("Predicted label distribution:", np.bincount(y_pred))
print("Classification report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

new_message = ["Win free iPhone by clicking this link"]
new_vector = vectorizer.transform(new_message)
predicted_class = model.predict(new_vector)
predicted_label = label_encoder.inverse_transform(predicted_class)[0]
print("Prediction for new message:", predicted_label)
