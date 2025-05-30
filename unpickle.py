import pickle

with open(r"processed_pdb\1b\11ba.pkl", "rb") as f:
    data = pickle.load(f)

print(type(data))
print(data)  # Or summarize using `len`, `keys()`, etc.
