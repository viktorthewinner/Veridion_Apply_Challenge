import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# === 1. Încarcă embeddings-urile și datele validate manual ===
X = np.load("saved_embeddings/company_embeddings.npy")

# Încarcă fișierul cu label-uri validate (primele 50 etichete corectate manual)
data = pd.read_csv("validated_data_first_50.csv")

# === 2. Selectează doar primele 50 (cele validate manual) ===
X_train = X[:50]
y_labels = data["insurance_label"].iloc[:50].apply(eval)  # evaluăm lista string în listă efectivă

# === 3. Transformă etichetele multiple în binar ===
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(y_labels)

# === 4. Antrenează un model scikit-learn (ex: RandomForest) ===
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# === 5. Evaluează pe un set de test (ex: următoarele 100) ===
X_test = X[50:150]
y_test_labels = data["insurance_label"].iloc[50:150].apply(eval)
y_test = mlb.transform(y_test_labels)

y_pred = clf.predict(X_test)

# === 6. Afișează performanța ===
print(classification_report(y_test, y_pred, target_names=mlb.classes_))
