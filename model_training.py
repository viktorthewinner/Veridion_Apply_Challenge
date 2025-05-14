import numpy as np
import pandas as pd
import os
import random
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
import joblib
import ast


# files
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

# i split my data in this 20%, 20%, 40%
# it is needed to add train data here as well, because i am addind the feedbacks and the model will train on them
# in this script there is a feedback loop, doing RLHF (human choose)

train_idx = list(range(51, 2000))
test_idx = list(range(2000, 4000))
new_data_idx = list(range(4000, len(X)))

X_train = X[train_idx]
Y_train = df.loc[train_idx, "insurance_label"]
X_test = X[test_idx]
Y_test = df.loc[test_idx, "insurance_label"]
X_new = X[new_data_idx]
new_df = df.loc[new_data_idx]

# add the data from feedback file

if os.path.exists(FEEDBACK_CSV):
    feedback_df = pd.read_csv(FEEDBACK_CSV)
    feedback_df["insurance_label"] = feedback_df["insurance_label"].apply(eval)
    X_feedback = []
    for i, row in feedback_df.iterrows():
        matching_idx = df[(df["description"] == row["description"]) & (df["business_tags"] == row["business_tags"])].index
        if len(matching_idx) > 0:
            idx = matching_idx[0]
            X_feedback.append(X[idx])

    if X_feedback:
        X_train = np.vstack([X_train] + X_feedback)
        Y_train = list(Y_train) + feedback_df["insurance_label"].tolist()

# deploy the model
# if there is no model (critical point), train it from zero

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

# saving the model
joblib.dump(classifier, model_path)
joblib.dump(mlb, mlb_path)

# start of the feedback loop; there are chosed 20 random cases and then the user can choose the right labels
# because it can take a while, maybe the user will want to stop after some training, so there is a 'stop' added
# RLHF (reinforcement learning from human feedback)

print("\n🔁 Starting feedback loop on 20 random 'new' cases with improved prediction handling...")
feedback_data = []

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
    top_indices = probs.argsort()[::-1][:3]

    top_labels = [mlb.classes_[i] for i in top_indices]

    # if no right labels, use the one from zero-shoot
    if not top_labels:
        initial_labels = df.loc[idx, "insurance_label"]
        if not isinstance(initial_labels, list):
            initial_labels = []
        top_labels = initial_labels
        if top_labels:
            print("⚠️ Model has no predicted labels.")
            print(f"🧠 Fallback to LLM generated labels: {top_labels}")
        else:
            print("⚠️ No fallback labels.")

    print("\n---")
    print(f"Description: {df.loc[idx, 'description']}")
    print(f"Business Tags: {df.loc[idx, 'business_tags']}")
    print(f"Sector: {df.loc[idx, 'sector']}")
    print(f"Category: {df.loc[idx, 'category']}")
    print(f"Niche: {df.loc[idx, 'niche']}")
    
    top_labels_str = [str(label) for label in top_labels]  
    print(f"Predicted Labels: {', '.join(top_labels_str)}")

    feedback = input("✅ Accept prediction? (y/n): ").strip().lower()

    if feedback == "n":
        correct_label_input = input("✏️ Enter correct labels as Python list (e.g., ['Life', 'Auto']): ")
        try:
            correct_label = ast.literal_eval(correct_label_input)
            if isinstance(correct_label, list) and all(isinstance(label, str) for label in correct_label):
                df.at[idx, 'insurance_label'] = correct_label
                feedback_data.append(df.loc[idx])
                X_train = np.vstack([X_train, embedding])
                Y_train_bin = np.vstack([Y_train_bin, mlb.transform([correct_label])])
                classifier.fit(X_train, Y_train_bin)
                print("✅ Model updated with corrected labels.")
            else:
                print("❌ Wrong format.")
        except Exception as e:
            print(f"❌ Skipping update. {e}")
    else:
        feedback_data.append(df.loc[idx])
        print("👍 Prediction accepted.")

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

joblib.dump(classifier, model_path)
joblib.dump(mlb, mlb_path)

print("\n✅ Script compiled successfully.")

# after each new update, there is a '.fit'
# each example is added to 'validated_data_feedback.csv'
