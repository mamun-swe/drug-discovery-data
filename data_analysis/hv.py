import pandas as pd

# Load the dataset
df = pd.read_csv("../data/virus_activity/raw/merged_all.csv", sep='\t')

# Filter rows for only "Human immunodeficiency virus 1"
hiv_df = df[df["virus_name"] == "Human immunodeficiency virus 1"]

# Display basic info and first few rows
print(f"Number of rows for Human immunodeficiency virus 1: {len(hiv_df)}")
print(hiv_df.head())

# Save filtered data to a new CSV file
output_path = "../data/virus_activity/filtered/HIV1_data.csv"
hiv_df.to_csv(output_path, sep='\t', index=False)

print(f"Filtered dataset saved to: {output_path}")
