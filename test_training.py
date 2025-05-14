import numpy as np
import pandas as pd
import os
import random
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
import joblib

EMBEDDING_PATH = "saved_embeddings/company_embeddings.npy"
LABELLED_CSV = "manual_validation_needed.csv"
FEEDBACK_CSV = "validated_data_feedback.csv"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

X = np.load(EMBEDDING_PATH)
df = pd.read_csv(LABELLED_CSV)
df["insurance_label"] = df["insurance_label"].apply(eval)
df = df[df["insurance_label"].map(len) > 0]
X = X[df.index]

train_idx = list(range(51, 2000))
test_idx = list(range(2000, 4000))
new_data_idx = list(range(4000, 9000))

X_train = X[train_idx]
Y_train = df.loc[train_idx, "insurance_label"]
X_test = X[test_idx]
Y_test = df.loc[test_idx, "insurance_label"]
X_new = X[new_data_idx]
new_df = df.loc[new_data_idx]

if os.path.exists(FEEDBACK_CSV):
    feedback_df = pd.read_csv(FEEDBACK_CSV)
    feedback_df["insurance_label"] = feedback_df["insurance_label"].apply(eval)
    X_feedback = []
    for i, row in feedback_df.iterrows():
        idx = df[(df["description"] == row["description"]) & (df["business_tags"] == row["business_tags"])].index
        if len(idx) > 0:
            X_feedback.append(X[idx[0]])
    if X_feedback:
        X_train = np.vstack([X_train] + X_feedback)
        Y_train = list(Y_train) + feedback_df["insurance_label"].tolist()

mlb_path = os.path.join(MODEL_DIR, "label_binarizer.pkl")
model_path = os.path.join(MODEL_DIR, "classifier.pkl")

if os.path.exists(model_path) and os.path.exists(mlb_path):
    classifier = joblib.load(model_path)
    mlb = joblib.load(mlb_path)
    Y_train_bin = mlb.transform(Y_train)
else:
    mlb = MultiLabelBinarizer()
    Y_train_bin = mlb.fit_transform(Y_train)
    classifier = OneVsRestClassifier(LogisticRegression(solver='liblinear', max_iter=1000))
    classifier.fit(X_train, Y_train_bin)

Y_test_bin = mlb.transform(Y_test)
Y_pred = classifier.predict(X_test)
report = classification_report(Y_test_bin, Y_pred, target_names=mlb.classes_, zero_division=0)
print(report)

joblib.dump(classifier, model_path)
joblib.dump(mlb, mlb_path)

print("\n🔁 Starting feedback loop on 20 random 'new' cases with improved prediction handling...")
feedback_data = []

# Asigură-te că insurance_label e listă
df["insurance_label"] = df["insurance_label"].apply(
    lambda x: eval(x) if isinstance(x, str) else x
)

sample_indices = random.sample(list(new_df.index), min(20, len(new_df)))

for idx in sample_indices:
    stop = input("⛔️ Type 'stop' to end feedback loop, or press Enter to continue: ").strip().lower()
    if stop == "stop":
        print("🔚 Feedback loop stopped by user.")
        break

    embedding = X[idx].reshape(1, -1)
    probs = classifier.predict_proba(embedding)[0]
    top_indices = probs.argsort()[::-1][:3]  # top 3 clase, indiferent de probabilitate

    # Obține etichetele corespunzătoare
    top_labels = [mlb.classes_[i] for i in top_indices]

    # Dacă nu există etichete, recurge la etichetele generate inițial
    if not top_labels:
        initial_labels = df.loc[idx, "insurance_label"]
        if not isinstance(initial_labels, list):
            initial_labels = []
        top_labels = initial_labels
        if top_labels:
            print("⚠️ Modelul nu a făcut o predicție de încredere.")
            print(f"🧠 Fallback la etichetele generate de LLM: {top_labels}")
        else:
            print("⚠️ Nicio etichetă disponibilă pentru fallback.")

    print("\n---")
    print(f"Description: {df.loc[idx, 'description']}")
    print(f"Business Tags: {df.loc[idx, 'business_tags']}")
    print(f"Sector: {df.loc[idx, 'sector']}")
    print(f"Category: {df.loc[idx, 'category']}")
    print(f"Niche: {df.loc[idx, 'niche']}")
    print(f"Predicted Labels: {top_labels}")

    feedback = input("✅ Accept prediction? (y/n): ").strip().lower()

    if feedback == "n":
        correct_label = input("✏️ Enter correct labels as Python list (e.g., ['Life', 'Auto']): ")
        try:
            correct_label = eval(correct_label)
            if isinstance(correct_label, list):
                df.loc[idx, 'insurance_label'] = correct_label
                feedback_data.append(df.loc[idx])
                X_train = np.vstack([X_train, embedding])
                Y_train_bin = np.vstack([Y_train_bin, mlb.transform([correct_label])])
                classifier.fit(X_train, Y_train_bin)
                print("✅ Model updated with corrected label.")
            else:
                print("❌ Etichetele nu sunt o listă. Ignorat.")
        except:
            print("❌ Format invalid. Skipping update.")
    else:
        feedback_data.append(df.loc[idx])
        print("👍 Prediction accepted.")

# 🔄 Salvare feedback
print("\n💾 Salvare feedback actualizat...")

if os.path.exists(FEEDBACK_CSV):
    existing_feedback = pd.read_csv(FEEDBACK_CSV)
    existing_feedback["insurance_label"] = existing_feedback["insurance_label"].apply(
        lambda x: eval(x) if isinstance(x, str) else x
    )
    combined = pd.concat([existing_feedback, pd.DataFrame(feedback_data)], ignore_index=True)
    combined.drop_duplicates(subset=["description", "business_tags"], inplace=True)
else:
    combined = pd.DataFrame(feedback_data)

combined.to_csv(FEEDBACK_CSV, index=False)

# 🔐 Salvare model updatat
joblib.dump(classifier, model_path)
joblib.dump(mlb, mlb_path)

print("\n✅ Totul este salvat. Feedback și model actualizate.")
