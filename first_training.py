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
MODEL_DIR = "model"
FEEDBACK_CSV = "validated_data_feedback.csv"
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

mlb = MultiLabelBinarizer()
Y_train_bin = mlb.fit_transform(Y_train)
Y_test_bin = mlb.transform(Y_test)

classifier = OneVsRestClassifier(LogisticRegression(solver='liblinear', max_iter=1000))
classifier.fit(X_train, Y_train_bin)

Y_pred = classifier.predict(X_test)
report = classification_report(Y_test_bin, Y_pred, target_names=mlb.classes_, zero_division=0)
print(report)

joblib.dump(classifier, os.path.join(MODEL_DIR, "classifier.pkl"))
joblib.dump(mlb, os.path.join(MODEL_DIR, "label_binarizer.pkl"))

print("\nStarting feedback loop on 20 random 'new' cases...")
feedback_data = []
sample_indices = random.sample(list(new_df.index), min(20, len(new_df)))

for idx in sample_indices:
    embedding = X[idx].reshape(1, -1)
    prediction = classifier.predict(embedding)
    labels = mlb.inverse_transform(prediction)[0]

    print("\n---")
    print(f"Description: {df.loc[idx, 'description']}")
    print(f"Business Tags: {df.loc[idx, 'business_tags']}")
    print(f"Predicted Labels: {labels}")
    
    feedback = input("Accept prediction? (y/n): ").strip().lower()
    
    if feedback == "n":
        correct_label = input("Enter correct labels as Python list (e.g., ['Life', 'Auto']): ")
        try:
            correct_label = eval(correct_label)
            df.loc[idx, 'insurance_label'] = correct_label
            feedback_data.append(df.loc[idx])
            X_train = np.vstack([X_train, embedding])
            Y_train_bin = np.vstack([Y_train_bin, mlb.transform([correct_label])])
            classifier.fit(X_train, Y_train_bin)
            print("Model updated with corrected label.")
        except:
            print("Invalid format. Skipping update.")
    else:
        feedback_data.append(df.loc[idx])
        print("Prediction accepted.")

print("\nSaving updated data with feedback...")

if os.path.exists(FEEDBACK_CSV):
    existing_feedback = pd.read_csv(FEEDBACK_CSV)
    existing_feedback["insurance_label"] = existing_feedback["insurance_label"].apply(eval)
    combined_feedback = pd.concat([existing_feedback, pd.DataFrame(feedback_data)], ignore_index=True)
    combined_feedback.drop_duplicates(subset=["description", "business_tags"], inplace=True)
else:
    combined_feedback = pd.DataFrame(feedback_data)

combined_feedback.to_csv(FEEDBACK_CSV, index=False)

print("\nAll done. Model and feedback loop complete.")
