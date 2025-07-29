import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 1. Charger et préparer les données
data = []
with open(r"C:\Users\Waad RTIBI\exerciceChatBot\train.txt", 'r', encoding='utf-8') as f:
    for line in f:
        if ';' in line:
            text, label = line.strip().split(';')
            data.append((text, label))

df = pd.DataFrame(data, columns=['text', 'label'])

le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])

# 2. Diviser le dataset
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label_encoded'], test_size=0.2, random_state=42)

# 3. TF-IDF vectorizer
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 4. Entraîner Logistic Regression
clf = LogisticRegression(max_iter=200)
clf.fit(X_train_tfidf, y_train)

# 5. Prédiction et évaluation
y_pred = clf.predict(X_test_tfidf)
print(classification_report(y_test, y_pred, target_names=le.classes_))

import pickle

# Sauvegarder le modèle ML
with open("model_ml.pkl", "wb") as f:
    pickle.dump(clf, f)

# Sauvegarder le TF-IDF
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

# Sauvegarder le label encoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
