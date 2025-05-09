import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

file = pd.read_csv("ml_insurance_challenge.csv")

sheet_id = "12ETd6-bxAfF-fNMMwSofzUuwlP2swai0w4OpG_-0xf0"
sheet_name = "insurance_taxonomy"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
tax = pd.read_csv(url)

# read all the data
# now, we can use a simple pre-trained model on NLP
# advantages: simple script, using zero-shot type of ML, some kind of sigmoid function for similary score
# disavantages: we may be having errors because of the lack of fine-tuning, tricky answers

model = SentenceTransformer('all-MiniLM-L6-v2')

def combined(row):
    fields = [str(row['description']), str(row['business_tags']), str(row['sector']), str(row['category']), str(row['niche'])]
    return ' | '.join(fields)

# because we are having strings, i took all the data and put them on one single vector
# the model that we are having is used for small phrases, not really ideal for us, but it can make a similarity score
# using the embeddings we are then calculating a similarity score (the higher the value, the better similarity) 
# (score is from -something to 1; 1 is the similarity of one word with himself)
# it is a slow approach, a matter of minutes for a basic answer

file['combined_text'] = file.apply(combined, axis=1)

company_embeddings = model.encode(file['combined_text'], convert_to_tensor=True, show_progress_bar=True)
tax_embeddings = model.encode(tax['label'], convert_to_tensor=True, show_progress_bar=True)

top_k = 3
insurance_labels = []

# for each companies, we are taking the most 3 relevant insurances
# problem here: we could try to set a score, but then we could have 2 critical points
#   1. when we dont have any label (set score is too high)
#   2. when we have more labels, some of them are irelevant (usually the lower scores of the picked items; set score is too low in this case)

for company_embedding in company_embeddings:
    cosine_scores = util.cos_sim(company_embedding, tax_embeddings)[0]
    top_results = torch.topk(cosine_scores, k=top_k)
    matched_labels = [tax['label'].iloc[idx.item()] for idx in top_results[1]]
    insurance_labels.append(matched_labels)

file['insurance_label'] = insurance_labels
file.to_csv("classified_file_sec_script.csv", index=False)
print(file.head())

# to compare my results with my main method, i saved the results in the file