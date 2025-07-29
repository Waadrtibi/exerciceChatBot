import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# 1. Charger les données
data = []
with open(r"C:\Users\Waad RTIBI\exerciceChatBot\train.txt", 'r', encoding='utf-8') as f:
    for line in f:
        if ';' in line:
            text, label = line.strip().split(';')
            data.append((text, label))

df = pd.DataFrame(data, columns=['text', 'label'])

# 2. Encoder les labels (émotions)
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])
num_classes = len(le.classes_)

# 3. Séparer en train/test
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label_encoded'], test_size=0.2, random_state=42)

# 4. Tokenization des textes
max_words = 5000      # Nombre max de mots dans le vocabulaire
max_len = 50          # Longueur max des séquences (padding/troncature)

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post')

# 5. One-hot encoding des labels pour la classification multi-classes
y_train_cat = to_categorical(y_train, num_classes=num_classes)
y_test_cat = to_categorical(y_test, num_classes=num_classes)

# 6. Construction du modèle LSTM
embedding_dim = 100

model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

# 7. Entraînement du modèle
history = model.fit(
    X_train_pad,
    y_train_cat,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# 8. Évaluation sur le jeu de test
loss, accuracy = model.evaluate(X_test_pad, y_test_cat, verbose=1)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# 9. Sauvegarder le modèle et le tokenizer
model.save('dl_lstm_model.h5')

import pickle
with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('label_encoder.pkl', 'wb') as handle:
    pickle.dump(le, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Sauvegarde du tokenizer
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Sauvegarde du label encoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# Sauvegarde du modèle LSTM
model.save("dl_lstm_model.h5")
