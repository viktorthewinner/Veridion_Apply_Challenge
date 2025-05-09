import pandas as pd

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


