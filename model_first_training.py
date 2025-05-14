import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import joblib
import ast
import os

# in this point, i have the embeddings for companies
# because i have the basic LLM, i want to fine-tune with the first 50 manual validated
# but because i got some errors (under-fitting), and because it is taking too long to do it by hand, i will add supervised learning (similar to RLHF)
# it is hard to minimize the number of training data, because there will be a lot of labels still unused

# files
CSV_FILE = "validated_data_first_50.csv"
EMBEDDING_FILE = "saved_embeddings/company_embeddings.npy"
CSV_FILE_2 = "manual_validation_needed.csv"  # Fișierul cu toate datele

# first batch to train is from validated data
df_first_batch = pd.read_csv(CSV_FILE)

# strings to list items
df_first_batch['insurance_label'] = df_first_batch['insurance_label'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)

# for training i have to use the embeddings
X_first_batch = np.load(EMBEDDING_FILE)
X_first_batch = X_first_batch[:50]

# reading all data
df_full = pd.read_csv(CSV_FILE_2)

# taking the rest of training data
df_second_batch = df_full.iloc[50:2000]

df_second_batch.loc[:, 'insurance_label'] = df_second_batch['insurance_label'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)

# embeddings again
X_second_batch = np.load(EMBEDDING_FILE)
X_second_batch = X_second_batch[50:2000] 

# make one vector for data, make one vector for embeddings
X_combined = np.concatenate([X_first_batch, X_second_batch], axis=0)
y_combined = pd.concat([df_first_batch['insurance_label'], df_second_batch['insurance_label']], axis=0)

# using multilabelbinazer (mlb will have 0/1 for every embedding)
mlb = MultiLabelBinarizer()
y_combined_bin = mlb.fit_transform(y_combined)

# using this model, i can use multiple labels
classifier = OneVsRestClassifier(LogisticRegression(solver='liblinear'))
classifier.fit(X_combined, y_combined_bin)

# saving the model and mlb
os.makedirs("model", exist_ok=True)
joblib.dump(classifier, "model/classifier.pkl")
joblib.dump(mlb, "model/label_binarizer.pkl")

print("✅ Model is trained on 50 + 1950 data!")

# so now i have a pre-trained model using embeddings
# the bad part of this is i zero-shoot nearly all the data, so there is a big inaccuracy
