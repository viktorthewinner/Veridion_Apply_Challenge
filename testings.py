import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

# Încarcă modelul și tokenizer-ul DistilBERT
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertModel.from_pretrained(model_name)

# Functie pentru a obține embeddings din text
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze()  # Luăm media vectorilor din ultimul strat
    return embedding.numpy()

# Încarcă fișierul cu date
df = pd.read_csv("ml_insurance_challenge.csv")
df["description"] = df["description"].astype(str)  # Asigură-te că textul este de tip string

# Obține embeddings pentru fiecare descriere
embeddings = np.array([get_embedding(text) for text in df["description"]])

# Calculăm similaritatea cosinus între primele două descriere (exemplu)
similarity = cosine_similarity([embeddings[0]], [embeddings[1]])
print("Cosine similarity între primele două descriere:", similarity[0][0])

# Salvăm embeddings într-un fișier pentru utilizare ulterioară
np.save("company_embeddings_distilbert.npy", embeddings)

# Poți adăuga o coloană cu etichetele de predicție (folosind modelul tău de predicție)
# Exemplu de aplicare a unui model de predicție și adăugare a predicției în dataframe
# În acest caz, adaug doar un exemplu generic de predicție
df["predicted_label"] = ["label_example" for _ in range(len(df))]  # Înlocuiește cu etichetele tale efective

# Salvează fișierul cu rezultatele
df.to_csv("final_data.csv", index=False)
