from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline

df = pd.read_csv("validated_companies.csv")
X = df["combined_text"]
y_raw = df["insurance_label"].apply(eval)  

mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(y_raw)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000)),
    ("clf", MultiOutputClassifier(LogisticRegression(max_iter=1000)))
])


pipeline.fit(X, Y)

X_new = companies["combined_text"]
Y_pred = pipeline.predict(X_new)
pred_labels = mlb.inverse_transform(Y_pred)
companies["ml_insurance_label"] = pred_labels


companies.to_csv("hybrid_classified_companies.csv", index=False)
