import pandas as pd
import numpy as np
import joblib

# files
MODEL_PATH = "model/classifier.pkl"
MLB_PATH = "model/label_binarizer.pkl"
EMBEDDING_PATH = "saved_embeddings/company_embeddings.npy"
INPUT_CSV = "ml_insurance_challenge.csv"
OUTPUT_CSV = "classified_companies.csv"

# because i have embeddings just for the clean data i have to drop the ones with missing info
df = pd.read_csv(INPUT_CSV)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

df.info()

X = np.load(EMBEDDING_PATH)

classifier = joblib.load(MODEL_PATH)
mlb = joblib.load(MLB_PATH)

probs_per_label = classifier.predict_proba(X)

try:
    y_probs = np.vstack([p[:, 1] for p in probs_per_label]).T  
except IndexError:
    y_probs = np.array([p[0] for p in probs_per_label]).reshape(-1, 1)
 
top3_labels = []
for probs in y_probs:
    top_indices = probs.argsort()[::-1][:3]
    labels = [mlb.classes_[i] for i in top_indices]
    top3_labels.append(', '.join(str(label) for label in labels)) 


print(len(top3_labels))

df['insurance_label'] = top3_labels

df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Predictions saved in '{OUTPUT_CSV}'")


# this is the last script, for final results
# its purpose comes after there is some training on the model
# if there is a new round of training, this script must be compiled again