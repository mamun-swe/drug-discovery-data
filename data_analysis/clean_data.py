#!/usr/bin/env python3
"""
Data Cleaning and Preparation Script for Antiviral Drug Discovery
==================================================================

This script performs:
1. Duplicate removal from the dataset
2. Virus name standardization
3. Dataset balancing (8000-8500 samples per virus)
4. SMILES validation
5. Feature engineering

Author: Shamsuddin Ahmed
Date: 2025-11-05
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import json


def load_data(filepath):
    """Load the CSV data"""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, sep='\t')
    print(f"✅ Loaded: {len(df):,} rows, {df.shape[1]} columns")
    return df


def standardize_virus_names(df):
    """Standardize virus names to merge duplicates"""
    print("\n🧹 Standardizing virus names...")

    virus_mapping = {
        # HIV variants
        'Human immunodeficiency virus 1': 'HIV-1',
        'Human immunodeficiency virus type 1': 'HIV-1',
        'Human immunodeficiency virus': 'HIV-1',
        'HIV-1': 'HIV-1',

        # SARS-CoV-2
        'Severe acute respiratory syndrome coronavirus 2': 'SARS-CoV-2',
        'SARS-CoV-2': 'SARS-CoV-2',

        # Influenza
        'Influenza A virus': 'Influenza A',
        'Influenza': 'Influenza A',
        'Influenza A': 'Influenza A',

        # Hepatitis C
        'Hepacivirus hominis': 'Hepatitis C',
        'HCV': 'Hepatitis C',
        'Hepatitis C virus': 'Hepatitis C',
        'Hepatitis C': 'Hepatitis C',

        # Dengue
        'dengue virus type 2': 'Dengue',
        'Dengue virus': 'Dengue',
        'Dengue': 'Dengue',

        # Zika
        'Zika virus': 'Zika',
        'Zika': 'Zika',

        # Hepatitis B
        'Hepatitis B virus': 'Hepatitis B',
        'HBV': 'Hepatitis B',
        'Hepatitis B': 'Hepatitis B',
    }

    df['virus_name_clean'] = df['virus_name'].map(virus_mapping).fillna(df['virus_name'])

    print("Virus counts AFTER standardization:")
    print(df['virus_name_clean'].value_counts().head(10))

    return df


def remove_duplicates(df):
    """Remove duplicate entries based on SMILES + virus"""
    print(f"\n🗑️  Removing duplicates...")
    print(f"Before: {len(df):,} rows")

    df_clean = df.drop_duplicates(subset=['smiles', 'virus_name_clean'], keep='first')

    removed = len(df) - len(df_clean)
    print(f"After: {len(df_clean):,} rows")
    print(f"Removed: {removed:,} duplicates ({removed / len(df) * 100:.2f}%)")

    return df_clean


def balance_dataset(df, top_n=6, target_samples=8000):
    """Select top N viruses and balance to target_samples each"""
    print(f"\n⚖️  Balancing dataset...")

    # Get top N viruses
    virus_counts = df['virus_name_clean'].value_counts()
    top_viruses = virus_counts.head(top_n).index.tolist()

    print(f"\nTop {top_n} viruses selected:")
    for i, virus in enumerate(top_viruses, 1):
        print(f"{i}. {virus}: {virus_counts[virus]:,} samples")

    # Filter to top viruses
    df_top = df[df['virus_name_clean'].isin(top_viruses)].copy()

    # Balance
    balanced_dfs = []
    for virus in top_viruses:
        virus_data = df_top[df_top['virus_name_clean'] == virus]

        if len(virus_data) >= target_samples:
            sampled = virus_data.sample(n=target_samples, random_state=42)
        else:
            sampled = virus_data

        balanced_dfs.append(sampled)
        print(f"  {virus}: {len(sampled):,} samples")

    df_balanced = pd.concat(balanced_dfs, ignore_index=True)
    print(f"\n✅ Balanced dataset: {len(df_balanced):,} total samples")

    return df_balanced, top_viruses


def validate_smiles(df):
    """Validate SMILES strings using RDKit"""
    print("\n🧪 Validating SMILES...")

    def is_valid_smiles(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False

    df['valid_smiles'] = df['smiles'].apply(is_valid_smiles)

    invalid_count = (~df['valid_smiles']).sum()
    print(f"Invalid SMILES: {invalid_count} ({invalid_count / len(df) * 100:.2f}%)")

    df_valid = df[df['valid_smiles']].copy()
    print(f"Valid dataset: {len(df_valid):,} samples")

    return df_valid


def create_activity_labels(df):
    """Create binary activity labels (active/inactive)"""
    print("\n🎯 Creating activity labels...")

    def create_label(row):
        try:
            value = float(row['activity_value'])
            unit = row['activity_unit']
            activity_type = row['activity_type']

            # Convert to nM
            if unit == 'uM':
                value_nm = value * 1000
            elif unit == 'nM':
                value_nm = value
            elif unit == 'mM':
                value_nm = value * 1000000
            elif unit == 'M':
                value_nm = value * 1000000000
            else:
                return np.nan

            # Active if IC50/Ki/EC50 < 10 μM (10000 nM)
            if activity_type in ['IC50', 'Ki', 'EC50', 'Kd']:
                return 1 if value_nm < 10000 else 0
            elif activity_type in ['Inhibition', '%']:
                return 1 if value > 50 else 0
            else:
                return np.nan
        except:
            return np.nan

    df['activity_label'] = df.apply(create_label, axis=1)
    df_final = df[df['activity_label'].notna()].copy()
    df_final['activity_label'] = df_final['activity_label'].astype(int)

    print(f"Final dataset: {len(df_final):,} samples")
    print(f"\nActivity distribution:")
    print(df_final['activity_label'].value_counts())
    print(f"Active ratio: {df_final['activity_label'].mean():.2%}")

    return df_final


def generate_fingerprints(df):
    """Generate Morgan fingerprints"""
    print("\n🔢 Generating Morgan fingerprints...")

    def smiles_to_fp(smiles, radius=2, n_bits=2048):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return np.zeros(n_bits)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            return np.array(fp)
        except:
            return np.zeros(n_bits)

    fingerprints = []
    for smiles in tqdm(df['smiles'], desc="Creating fingerprints"):
        fp = smiles_to_fp(smiles)
        fingerprints.append(fp)

    df['morgan_fp'] = fingerprints
    print("✅ Fingerprints generated")

    return df


def save_results(df, top_viruses, output_dir='.'):
    """Save cleaned dataset and summary"""
    print("\n💾 Saving results...")

    # Save main dataset
    output_file = f"{output_dir}/cleaned_balanced_dataset.csv"
    df.to_csv(output_file, index=False)
    print(f"✅ Saved: {output_file}")

    # Create summary
    summary = {
        'total_samples': len(df),
        'num_viruses': df['virus_name_clean'].nunique(),
        'viruses': top_viruses,
        'active_compounds': int(df['activity_label'].sum()),
        'inactive_compounds': int((df['activity_label'] == 0).sum()),
        'active_ratio': float(df['activity_label'].mean()),
        'virus_distribution': df['virus_name_clean'].value_counts().to_dict()
    }

    summary_file = f"{output_dir}/dataset_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Saved: {summary_file}")

    return summary


def main():
    """Main execution function"""
    print("=" * 70)
    print("🧬 ANTIVIRAL DRUG DISCOVERY - DATA CLEANING PIPELINE")
    print("=" * 70)

    # Configuration
    INPUT_FILE = '../data/virus_activity/raw/merged_all.csv'
    TOP_N_VIRUSES = 6
    TARGET_SAMPLES = 8000
    OUTPUT_DIR = '../data/virus_activity/'

    # Step 1: Load data
    df = load_data(INPUT_FILE)

    # Step 2: Standardize virus names
    df = standardize_virus_names(df)

    # Step 3: Remove duplicates
    df = remove_duplicates(df)

    # Step 4: Balance dataset
    df, top_viruses = balance_dataset(df, TOP_N_VIRUSES, TARGET_SAMPLES)

    # Step 5: Validate SMILES
    df = validate_smiles(df)

    # Step 6: Create activity labels
    df = create_activity_labels(df)

    # Step 7: Generate fingerprints
    df = generate_fingerprints(df)

    # Step 8: Save results
    summary = save_results(df, top_viruses, OUTPUT_DIR)

    # Final summary
    print("\n" + "=" * 70)
    print("📊 FINAL SUMMARY")
    print("=" * 70)
    print(json.dumps(summary, indent=2))
    print("=" * 70)
    print("\n✅ Data cleaning complete! Ready for model training.")


if __name__ == "__main__":
    main()
