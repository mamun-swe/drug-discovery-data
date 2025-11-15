"""
BALANCED VIRUS-BASED COMPOUND FILTER - FIXED VERSION
=====================================================
- Keeps top N viruses
- EXACTLY 50/50 active/inactive split for each virus
- ALL viruses have the SAME total compound count
- Uses CORRECT classification logic for different activity types
"""

import pandas as pd
import os

# ============================================================================
# CONFIGURATION - EDIT THESE VALUES
# ============================================================================
INPUT_FILE = "../data/virus_activity/raw/merged_all.csv"  # Your input CSV file
OUTPUT_FILE = "../data/virus_activity/filtered/main_dataset.csv"  # Output filename
TOP_N_VIRUSES = 2  # How many TOP viruses to keep (by compound count)
ACTIVITY_THRESHOLD = 10000  # nM - Standard: 10,000 nM (10 µM) for active/inactive


# ============================================================================
# SCRIPT - NO NEED TO EDIT BELOW THIS LINE
# ============================================================================

def classify_activity(row):
    """
    Classify compound as active or inactive based on activity value and type
    THIS MATCHES YOUR ANALYSIS SCRIPT EXACTLY!
    """
    activity_type = row['activity_type']
    activity_value = row['activity_value']

    # Handle different activity types
    if activity_type in ['IC50', 'EC50', 'Ki', 'Kd']:
        # Lower values = more active (more potent)
        return activity_value < ACTIVITY_THRESHOLD
    elif 'Inhibition' in str(activity_type):
        # Higher percentage = more active
        return activity_value > 50
    else:
        # Default: use threshold
        return activity_value < ACTIVITY_THRESHOLD


print("=" * 70)
print("BALANCED VIRUS-BASED COMPOUND FILTER - FIXED")
print("=" * 70)

print(f"\nReading {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE, sep='\t')
print(f"✓ Loaded {len(df)} rows")
print(f"✓ Unique compounds: {df['compound_id'].nunique()}")
print(f"✓ Unique SMILES: {df['smiles'].nunique()}")
print(f"✓ Unique viruses: {df['virus_name'].nunique()}")
print(f"\nActivity types in data: {df['activity_type'].unique()}")

# ============================================================================
# STEP 1: Remove duplicate SMILES (keep best/lowest activity per SMILES)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 1: Removing duplicate SMILES")
print("=" * 70)

# Keep the row with the lowest (best) activity value for each SMILES
df_unique = df.loc[df.groupby('smiles')['activity_value'].idxmin()].reset_index(drop=True)

print(f"✓ Removed {len(df) - len(df_unique)} duplicate SMILES")
print(f"✓ Remaining rows: {len(df_unique)}")

# ============================================================================
# STEP 2: Identify TOP N viruses by compound count
# ============================================================================
print("\n" + "=" * 70)
print(f"STEP 2: Selecting TOP {TOP_N_VIRUSES} viruses by compound count")
print("=" * 70)

virus_counts = df_unique.groupby('virus_name').size().sort_values(ascending=False)
print("\nVirus ranking by compound count:")
for i, (virus, count) in enumerate(virus_counts.items(), 1):
    marker = "✓ SELECTED" if i <= TOP_N_VIRUSES else "✗ excluded"
    print(f"  {i}. {virus}: {count} compounds [{marker}]")

# Select top N viruses
top_viruses = virus_counts.head(TOP_N_VIRUSES).index.tolist()
df_filtered = df_unique[df_unique['virus_name'].isin(top_viruses)].copy()

print(f"\n✓ Keeping {len(top_viruses)} viruses")
print(f"✓ Total compounds in selected viruses: {len(df_filtered)}")

# ============================================================================
# STEP 3: Classify compounds as active or inactive (CORRECT METHOD!)
# ============================================================================
print("\n" + "=" * 70)
print(f"STEP 3: Classifying compounds (CORRECT classification)")
print("=" * 70)
print(f"  IC50/EC50/Ki/Kd < {ACTIVITY_THRESHOLD} nM = Active")
print(f"  Inhibition > 50% = Active")

# Apply the CORRECT classification function
df_filtered['is_active'] = df_filtered.apply(classify_activity, axis=1)

print("\nClassification by virus:")
for virus in top_viruses:
    virus_data = df_filtered[df_filtered['virus_name'] == virus]
    n_active = virus_data['is_active'].sum()
    n_inactive = (~virus_data['is_active']).sum()
    print(f"  {virus}:")
    print(f"    Total: {len(virus_data)} | Active: {n_active} | Inactive: {n_inactive}")

# ============================================================================
# STEP 4: Find minimum compound count across viruses
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4: Finding minimum compound count for balancing")
print("=" * 70)

virus_compound_counts = {}
virus_active_counts = {}
virus_inactive_counts = {}

for virus in top_viruses:
    virus_data = df_filtered[df_filtered['virus_name'] == virus]
    active_data = virus_data[virus_data['is_active']]
    inactive_data = virus_data[~virus_data['is_active']]

    virus_compound_counts[virus] = len(virus_data)
    virus_active_counts[virus] = len(active_data)
    virus_inactive_counts[virus] = len(inactive_data)

print("\nAvailable compounds per virus:")
for virus in top_viruses:
    print(f"  {virus}:")
    print(f"    Total: {virus_compound_counts[virus]} | "
          f"Active: {virus_active_counts[virus]} | "
          f"Inactive: {virus_inactive_counts[virus]}")

# Find the minimum available for 50/50 split
min_active_available = min(virus_active_counts.values())
min_inactive_available = min(virus_inactive_counts.values())

# Calculate the balanced count (must be even for exact 50/50)
max_possible_per_category = min(min_active_available, min_inactive_available)
compounds_per_virus_per_category = max_possible_per_category
total_per_virus = compounds_per_virus_per_category * 2

print(f"\n🎯 Balancing strategy:")
print(f"  Minimum active available: {min_active_available}")
print(f"  Minimum inactive available: {min_inactive_available}")
print(f"  → Will keep {compounds_per_virus_per_category} active + {compounds_per_virus_per_category} inactive")
print(f"  → Total per virus: {total_per_virus} compounds (EXACT 50/50 split)")

# ============================================================================
# STEP 5: Create balanced dataset with EXACT 50/50 split
# ============================================================================
print("\n" + "=" * 70)
print("STEP 5: Creating balanced dataset (EXACT 50/50 per virus)")
print("=" * 70)

all_filtered = []

for virus in top_viruses:
    virus_data = df_filtered[df_filtered['virus_name'] == virus].copy()

    # Separate active and inactive
    active_data = virus_data[virus_data['is_active']].copy()
    inactive_data = virus_data[~virus_data['is_active']].copy()

    # Sort by activity (best first = lowest values for IC50/Ki, highest for Inhibition)
    active_data = active_data.sort_values('activity_value')
    inactive_data = inactive_data.sort_values('activity_value')

    # Keep EXACTLY the balanced amount
    active_kept = active_data.head(compounds_per_virus_per_category)
    inactive_kept = inactive_data.head(compounds_per_virus_per_category)

    # Combine
    virus_filtered = pd.concat([active_kept, inactive_kept])
    all_filtered.append(virus_filtered)

    # Calculate actual percentages
    actual_active_pct = (len(active_kept) / len(virus_filtered)) * 100
    actual_inactive_pct = (len(inactive_kept) / len(virus_filtered)) * 100

    print(f"\n{virus}:")
    print(f"  Before: {len(virus_data)} compounds ({len(active_data)} active, {len(inactive_data)} inactive)")
    print(f"  After:  {len(virus_filtered)} compounds ({len(active_kept)} active, {len(inactive_kept)} inactive)")
    print(f"  Split:  {actual_active_pct:.2f}% active, {actual_inactive_pct:.2f}% inactive")
    print(f"  Deleted: {len(virus_data) - len(virus_filtered)} compounds")

# Combine all filtered data
final_df = pd.concat(all_filtered, ignore_index=True)

# Remove the is_active column before saving
final_df = final_df.drop('is_active', axis=1)

# ============================================================================
# STEP 6: Save results
# ============================================================================
print("\n" + "=" * 70)
print("STEP 6: Saving results")
print("=" * 70)

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# Save to file
final_df.to_csv(OUTPUT_FILE, sep='\t', index=False)

print(f"\n✅ SUCCESS!")
print(f"✓ Original data: {len(df)} rows")
print(f"✓ After removing duplicate SMILES: {len(df_unique)} rows")
print(f"✓ Final filtered data: {len(final_df)} rows")
print(f"✓ Saved to: {OUTPUT_FILE}")

# ============================================================================
# SUMMARY STATISTICS (Using CORRECT classification)
# ============================================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY - PERFECTLY BALANCED DATASET")
print("=" * 70)

print(f"\nConfiguration:")
print(f"  Top viruses kept: {TOP_N_VIRUSES}")
print(f"  Activity threshold: {ACTIVITY_THRESHOLD} nM")
print(f"  Compounds per virus: {total_per_virus} (SAME for ALL viruses)")
print(f"  Split per virus: EXACTLY 50% active / 50% inactive")

print(f"\nFinal dataset:")
print(f"  Total rows: {len(final_df)}")
print(f"  Unique compounds: {final_df['compound_id'].nunique()}")
print(f"  Unique SMILES: {final_df['smiles'].nunique()}")
print(f"  Viruses: {final_df['virus_name'].nunique()}")

print(f"\n✓ VERIFICATION - Each virus has:")
print(f"  Total: {total_per_virus} compounds")
print(f"  Active: {compounds_per_virus_per_category} compounds (50.00%)")
print(f"  Inactive: {compounds_per_virus_per_category} compounds (50.00%)")

print("\nBreakdown by virus (using CORRECT classification):")
for virus in final_df['virus_name'].unique():
    virus_subset = final_df[final_df['virus_name'] == virus].copy()
    # Recalculate is_active using CORRECT method
    virus_subset['is_active_check'] = virus_subset.apply(classify_activity, axis=1)
    n_active = virus_subset['is_active_check'].sum()
    n_inactive = (~virus_subset['is_active_check']).sum()
    pct_active = (n_active / len(virus_subset)) * 100
    pct_inactive = (n_inactive / len(virus_subset)) * 100
    print(f"  {virus}:")
    print(
        f"    Total: {len(virus_subset)} | Active: {n_active} ({pct_active:.2f}%) | Inactive: {n_inactive} ({pct_inactive:.2f}%)")

print("\n" + "=" * 70)
print("✅ ALL VIRUSES NOW HAVE EXACT 50/50 SPLIT AND SAME COUNT!")
print("=" * 70)
