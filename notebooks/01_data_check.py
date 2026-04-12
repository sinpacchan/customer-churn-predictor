import pandas as pd

df = pd.read_csv("data/churn.csv")

print("Shape:", df.shape)
print("\nFirst rows:")
print(df.head())