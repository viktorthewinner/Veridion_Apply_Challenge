import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
import os

save_dir = "saved_embeddings"
os.makedirs(save_dir, exist_ok=True)

# folder in which i save embeddings

data_file = pd.read_csv("ml_insurance_challenge.csv")
print(data_file.head())

sheet_id = "12ETd6-bxAfF-fNMMwSofzUuwlP2swai0w4OpG_-0xf0"
sheet_name = "insurance_taxonomy"

url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"

data_sheet = pd.read_csv(url)
print(data_sheet.head())

# read all the data and printing a few to be sure
# i made a copy and clear the null data, they are not useful in the learning process, because they can give bad feedback

data_exp = data_file.copy()
data_exp.info()
data_exp.dropna(inplace=True)
data_exp.info()


def combined(row):
    fields = [str(row['description']), str(row['business_tags']), str(row['sector']), str(row['category']), str(row['niche'])]
    return ' | '.join(fields)

all_data = data_exp.apply(combined, axis=1)

# we have a vector just for rows named 'all_data'
# using the idea from the other script, i join all the strings together for semantic value
# will use this pre-defined LLM

model = SentenceTransformer("all-MiniLM-L6-v2")

company_embeddings = model.encode(all_data.tolist(), convert_to_tensor=True, show_progress_bar=True)
tax_embeddings = model.encode(data_sheet["label"].tolist(), convert_to_tensor=True, show_progress_bar=True)

insurance_labels = []
for company_embedding in company_embeddings:
    scores = util.cos_sim(company_embedding, tax_embeddings)[0]
    top_results = torch.topk(scores, k=3)
    top_labels = [data_sheet["label"].iloc[int(i)] for i in top_results.indices]
    insurance_labels.append(top_labels)

data_exp["insurance_label"]= insurance_labels

data_exp.to_csv("manual_validation_needed.csv", index=False)

# similar to second script, but this time the data will be corrected
# after manual validation, model will ajust
# took the first 50 tags and validated/changed

np.save(os.path.join(save_dir, "company_embeddings.npy"), company_embeddings.cpu().numpy())
np.save(os.path.join(save_dir, "taxonomy_embeddings.npy"), tax_embeddings.cpu().numpy())


# save the model
model_save_path = os.path.join(save_dir, "sentence_transformer_model")
model.save(model_save_path)

print(f"Model saved to {model_save_path}")