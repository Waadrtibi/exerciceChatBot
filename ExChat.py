import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 1. Lire le fichier train.txt
data = []
with open(r"C:\Users\Waad RTIBI\exerciceChatBot\train.txt", 'r', encoding='utf-8') as f:
    for line in f:
        if ';' in line:
            text, label = line.strip().split(';')
            data.append((text, label))

# 2. Créer un DataFrame
df = pd.DataFrame(data, columns=['text', 'label'])

# 3. Encoder les labels (émotions)
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])

print(df.head())

# Afficher les classes encodées
print("Classes:", list(le.classes_))
