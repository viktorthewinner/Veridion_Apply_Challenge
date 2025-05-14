import numpy as np
import pandas as pd
import os
import random
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
import joblib
import ast  # Import pentru evaluarea sigura a listelor
# Setări directoare și fișiere
EMBEDDING_PATH = "saved_embeddings/company_embeddings.npy"
LABELLED_CSV = "manual_validation_needed.csv"
FEEDBACK_CSV = "validated_data_feedback.csv"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)


# Încarcă modelul sau creează unul nou
mlb_path = os.path.join(MODEL_DIR, "label_binarizer.pkl")
model_path = os.path.join(MODEL_DIR, "classifier.pkl")

if os.path.exists(model_path) and os.path.exists(mlb_path):
    classifier = joblib.load(model_path)
    mlb = joblib.load(mlb_path)
else:
    mlb = MultiLabelBinarizer()

print("Number of classes:", len(mlb.classes_))
print("Classes:", mlb.classes_)
