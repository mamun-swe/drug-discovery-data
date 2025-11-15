import pandas as pd

# Load CSV
df = pd.read_csv("../data/virus_activity/raw/merged_all.csv", sep='\t')

# Find top 10 viruses by frequency
top_10_viruses = df["virus_name"].value_counts().head(10).index

# Filter dataset to include only those top 10 viruses
df_top10 = df[df["virus_name"].isin(top_10_viruses)]

# Find the smallest count among them
min_count = df_top10["virus_name"].value_counts().min()

# Sample equal number of rows per virus
balanced_df = (
    df_top10.groupby("virus_name", group_keys=False)
    .sample(n=min_count, random_state=42)
    .reset_index(drop=True)
)

# Save balanced dataset
balanced_df.to_csv("../data/virus_activity/cleaned_balanced_top10.csv", sep='\t', index=False)

print("✅ Balanced dataset created successfully!")
print(f"Each virus has {min_count} records.")
print("Saved to: ../data/virus_activity/cleaned_balanced_top10.csv")
print("\nVirus counts after balancing:")
print(balanced_df["virus_name"].value_counts())
