import os
import pandas as pd

os.makedirs("data", exist_ok=True)

df = pd.read_csv("data/Norm_Fused_Dataset.csv")

# 10k–30k is good for Streamlit Cloud
sample = df.sample(n=20000, random_state=42)

sample.to_csv("data/sample.csv", index=False)
print("Saved data/sample.csv:", sample.shape)
