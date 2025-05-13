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

print("\n🔁 Starting feedback loop on 20 random 'new' cases...")
feedback_data = []
sample_indices = random.sample(list(new_df.index), min(20, len(new_df)))

for idx in sample_indices:
    stop = input("⛔️ Type 'stop' to end feedback loop, or press Enter to continue: ").strip().lower()
    if stop == "stop":
        print("🔚 Feedback loop stopped by user.")
        break

    embedding = X[idx].reshape(1, -1)
    prediction = classifier.predict(embedding)
    labels = mlb.inverse_transform(prediction)[0]

    print("\n---")
    print(f"Description: {df.loc[idx, 'description']}")
    print(f"Business Tags: {df.loc[idx, 'business_tags']}")
    print(f"Sector: {df.loc[idx, 'sector']}")
    print(f"Category: {df.loc[idx, 'category']}")
    print(f"Niche: {df.loc[idx, 'niche']}")
    print(f"Predicted Labels: {labels}")
    
    feedback = input("✅ Accept prediction? (y/n): ").strip().lower()
    
    if feedback == "n":
        correct_label = input("✏️ Enter correct labels as Python list (e.g., ['Life', 'Auto']): ")
        try:
            correct_label = eval(correct_label)
            df.loc[idx, 'insurance_label'] = correct_label
            feedback_data.append(df.loc[idx])
            X_train = np.vstack([X_train, embedding])
            Y_train_bin = np.vstack([Y_train_bin, mlb.transform([correct_label])])
            classifier.fit(X_train, Y_train_bin)
            print("✅ Model updated with corrected label.")
        except:
            print("❌ Invalid format. Skipping update.")
    else:
        feedback_data.append(df.loc[idx])
        print("👍 Prediction accepted.")

print("\nSaving updated feedback...")

if os.path.exists(FEEDBACK_CSV):
    existing_feedback = pd.read_csv(FEEDBACK_CSV)
    existing_feedback["insurance_label"] = existing_feedback["insurance_label"].apply(eval)
    combined = pd.concat([existing_feedback, pd.DataFrame(feedback_data)], ignore_index=True)
    combined.drop_duplicates(subset=["description", "business_tags"], inplace=True)
else:
    combined = pd.DataFrame(feedback_data)

combined.to_csv(FEEDBACK_CSV, index=False)

joblib.dump(classifier, model_path)
joblib.dump(mlb, mlb_path)

print("\n✅ All done. Model and feedback saved.")
