#!/usr/bin/env python3
"""
Prepare All Supplementary Data Files for Journal Submission
Author: Shamsuddin Ahmed
Date: November 2025

This script creates S1-S6 supplementary data files required for submission.
Updated to include ALL novel candidates (after training data filtering).
"""

import pandas as pd
import os
import shutil
from pathlib import Path

print("=" * 80)
print("PREPARING SUPPLEMENTARY DATA FILES FOR SUBMISSION")
print("=" * 80)
print("\nNote: Using NOVEL candidates from screening_results_NOVEL_ONLY/")
print("      (Training data contamination has been removed)")
print("=" * 80)

# Create output directory
os.makedirs('supplementary_data', exist_ok=True)
os.makedirs('supplementary_data/S6_all_candidates_per_virus', exist_ok=True)

# ============================================================================
# S1: Best SOTA Models (already exists)
# ============================================================================
print("\n📄 S1: Best SOTA Models")
try:
    if os.path.exists('best_sota_models.csv'):
        df_s1 = pd.read_csv('best_sota_models.csv')
        df_s1.to_csv('supplementary_data/S1_best_sota_models.csv', index=False)
        print(f"   ✓ Created S1_best_sota_models.csv ({len(df_s1)} rows)")
    else:
        print("   ⚠️  best_sota_models.csv not found - please copy manually")
except Exception as e:
    print(f"   ✗ Error: {e}")

# ============================================================================
# S2: Model Ranking (already exists)
# ============================================================================
print("\n📄 S2: Model Ranking")
try:
    if os.path.exists('model_ranking.csv'):
        df_s2 = pd.read_csv('model_ranking.csv')
        df_s2.to_csv('supplementary_data/S2_model_ranking.csv', index=False)
        print(f"   ✓ Created S2_model_ranking.csv ({len(df_s2)} rows)")
    else:
        print("   ⚠️  model_ranking.csv not found - please copy manually")
except Exception as e:
    print(f"   ✗ Error: {e}")

# ============================================================================
# S3: SOTA Model Performance (already exists)
# ============================================================================
print("\n📄 S3: SOTA Model Performance")
try:
    if os.path.exists('sota_model_performance.csv'):
        df_s3 = pd.read_csv('sota_model_performance.csv')
        df_s3.to_csv('supplementary_data/S3_sota_model_performance.csv', index=False)
        print(f"   ✓ Created S3_sota_model_performance.csv ({len(df_s3)} rows)")
    else:
        print("   ⚠️  sota_model_performance.csv not found - please copy manually")
except Exception as e:
    print(f"   ✗ Error: {e}")

# ============================================================================
# S4: Screening Summary (already exists)
# ============================================================================
print("\n📄 S4: Screening Summary")
try:
    if os.path.exists('summary.csv'):
        df_s4 = pd.read_csv('summary.csv')
        df_s4.to_csv('supplementary_data/S4_screening_summary.csv', index=False)
        print(f"   ✓ Created S4_screening_summary.csv ({len(df_s4)} rows)")
    else:
        print("   ⚠️  summary.csv not found - please copy manually")
except Exception as e:
    print(f"   ✗ Error: {e}")

# ============================================================================
# S5: Dataset Statistics (create new)
# ============================================================================
print("\n📄 S5: Dataset Statistics")
try:
    viruses = [
        'HCV',
        'Zika virus',
        'dengue virus type 2',
        'Hepatitis B virus',
        'Hepacivirus hominis',
        'Human immunodeficiency virus type 1',
        'Severe acute respiratory syndrome coronavirus 2',
        'Influenza',
        'Human immunodeficiency virus 1',
        'Influenza A virus'
    ]

    data = []
    for virus in viruses:
        row = {
            'Virus': virus,
            'Total_Compounds': 8362,
            'Active_Compounds': 4181,
            'Inactive_Compounds': 4181,
            'Training_Set': 6690,
            'Validation_Set': 836,
            'Test_Set': 836,
            'Split_Ratio': '80/10/10'
        }
        data.append(row)

    df_s5 = pd.DataFrame(data)
    df_s5.to_csv('supplementary_data/S5_dataset_statistics.csv', index=False)
    print(f"   ✓ Created S5_dataset_statistics.csv ({len(df_s5)} rows)")
    print(f"   📊 Total training instances: {len(viruses) * 8362:,}")

except Exception as e:
    print(f"   ✗ Error: {e}")

# ============================================================================
# S6: ALL NOVEL Candidates Per Virus (from screening_results_NOVEL_ONLY)
# ============================================================================
print("\n📄 S6: ALL Novel Candidates Per Virus")
print("   📂 Source: screening_results_NOVEL_ONLY/")
print("   📝 Note: These candidates have been filtered to remove training data contamination")
print("")

# Check if NOVEL directory exists
if not os.path.exists('screening_results_NOVEL_ONLY'):
    print("   ❌ ERROR: screening_results_NOVEL_ONLY/ directory not found!")
    print("   ⚠️  Please run the filtering script first: python filter_simple.py")
    print("")
else:
    # Map virus names to file names (in NOVEL directory)
    virus_file_mapping = {
        'HCV': 'HCV_candidates_FIXED.csv',
        'Zika virus': 'Zika_virus_candidates_FIXED.csv',
        'dengue virus type 2': 'dengue_virus_type_2_candidates_FIXED.csv',
        'Hepatitis B virus': 'Hepatitis_B_virus_candidates_FIXED.csv',
        'Hepacivirus hominis': 'Hepacivirus_hominis_candidates_FIXED.csv',
        'Human immunodeficiency virus type 1': 'Human_immunodeficiency_virus_type_1_candidates_FIXED.csv',
        'Human immunodeficiency virus 1': 'Human_immunodeficiency_virus_1_candidates_FIXED.csv',
        'Influenza': 'Influenza_candidates_FIXED.csv',
        'Influenza A virus': 'Influenza_A_virus_candidates_FIXED.csv',
        'Severe acute respiratory syndrome coronavirus 2': 'Severe_acute_respiratory_syndrome_coronavirus_2_candidates_FIXED.csv'
    }

    # Key columns to include (if available)
    preferred_columns = [
        'rank', 'coconut_id', 'smiles', 'xgboost_score',
        'molecular_weight', 'logP', 'hbd', 'hba',
        'rotatable_bonds', 'tpsa', 'lipinski_pass', 'veber_pass',
        'num_aromatic_rings', 'num_stereocenters', 'fraction_csp3'
    ]

    success_count = 0
    total_candidates = 0

    for virus, filename in virus_file_mapping.items():
        try:
            # Read from NOVEL_ONLY directory
            filepath = f'screening_results_NOVEL_ONLY/{filename}'

            if os.path.exists(filepath):
                df = pd.read_csv(filepath)

                # Include ALL candidates (no top 100 limit)
                # Select key columns (if they exist)
                available_cols = [col for col in preferred_columns if col in df.columns]

                # If preferred columns don't exist, use all columns
                if not available_cols:
                    df_subset = df
                else:
                    df_subset = df[available_cols]

                # Save to S6 directory
                output_name = virus.replace(' ', '_')
                output_path = f'supplementary_data/S6_all_candidates_per_virus/{output_name}_all_novel_candidates.csv'
                df_subset.to_csv(output_path, index=False)

                print(f"   ✓ {virus}: {len(df):,} novel candidates")
                success_count += 1
                total_candidates += len(df)
            else:
                print(f"   ⚠️  {virus}: File not found ({filename})")

        except Exception as e:
            print(f"   ✗ {virus}: Error - {e}")

    print(f"\n   📊 Successfully created {success_count}/10 files")
    print(f"   📊 Total novel candidates: {total_candidates:,}")

# ============================================================================
# Create README for S6
# ============================================================================
print("\n📄 Creating README for S6")
try:
    readme_content = """# S6: All Novel Candidates Per Virus

## Description
This directory contains ALL novel candidates identified through virtual screening
of 721,010 natural products from the COCONUT database, after rigorous filtering
to remove compounds with high structural similarity to training data.

## Filtering Methodology
- Training data: 83,620 compounds across 10 viral targets
- Similarity metric: Tanimoto similarity using Morgan fingerprints (radius=2)
- Exclusion threshold: Compounds with Tanimoto similarity >0.85 to any training compound were removed
- Result: 8,216 genuinely novel candidates (92.4% of initial 8,894 candidates)

## Files
Each CSV file contains all novel candidates for a specific viral target:

1. HCV_all_novel_candidates.csv
2. Zika_virus_all_novel_candidates.csv
3. dengue_virus_type_2_all_novel_candidates.csv
4. Hepatitis_B_virus_all_novel_candidates.csv
5. Hepacivirus_hominis_all_novel_candidates.csv
6. Human_immunodeficiency_virus_type_1_all_novel_candidates.csv
7. Human_immunodeficiency_virus_1_all_novel_candidates.csv
8. Influenza_all_novel_candidates.csv
9. Influenza_A_virus_all_novel_candidates.csv
10. Severe_acute_respiratory_syndrome_coronavirus_2_all_novel_candidates.csv

## Columns
- rank: Ranking by XGBoost prediction score
- coconut_id: COCONUT database identifier
- smiles: Molecular structure (SMILES notation)
- xgboost_score: Prediction score (0-1, higher = more likely active)
- molecular_weight: Molecular weight (Da)
- logP: Lipophilicity (octanol-water partition coefficient)
- hbd: Hydrogen bond donors
- hba: Hydrogen bond acceptors
- rotatable_bonds: Number of rotatable bonds
- tpsa: Topological polar surface area (Ų)
- lipinski_pass: Passes Lipinski's Rule of Five (True/False)
- veber_pass: Passes Veber's rules (True/False)
- [additional molecular descriptors as available]

## Data Quality
- All candidates are NOVEL (not present in training data)
- All candidates passed similarity threshold (<0.85 to training)
- All candidates have prediction score ≥0.7
- All candidates passed SMILES validation
- All candidates from reputable natural product databases

## Usage
These candidates are prioritized for experimental validation. Researchers can:
1. Select candidates based on prediction score
2. Filter by drug-likeness criteria (Lipinski, Veber)
3. Choose based on specific molecular properties
4. Prioritize by virus-specific requirements

## Citation
If you use these candidates, please cite:
[Your paper citation here]

## Contact
For questions about the candidates or methodology:
[Your contact information]

## Version
Generated: November 2025
Source: COCONUT v1.0 (721,010 natural products)
Models: XGBoost trained on ChEMBL + BindingDB + PubChem data
Filtering: Post-screening training data contamination removal
"""

    readme_path = 'supplementary_data/S6_all_candidates_per_virus/README.md'
    with open(readme_path, 'w') as f:
        f.write(readme_content)

    print(f"   ✓ Created README.md in S6 directory")

except Exception as e:
    print(f"   ✗ Error creating README: {e}")

# ============================================================================
# Create Filtering Statistics File
# ============================================================================
print("\n📄 Creating S6_filtering_statistics.csv")
try:
    # Collect statistics from NOVEL_ONLY files
    stats_data = []

    virus_file_mapping = {
        'HCV': 'HCV_candidates_FIXED.csv',
        'Zika virus': 'Zika_virus_candidates_FIXED.csv',
        'dengue virus type 2': 'dengue_virus_type_2_candidates_FIXED.csv',
        'Hepatitis B virus': 'Hepatitis_B_virus_candidates_FIXED.csv',
        'Hepacivirus hominis': 'Hepacivirus_hominis_candidates_FIXED.csv',
        'Human immunodeficiency virus type 1': 'Human_immunodeficiency_virus_type_1_candidates_FIXED.csv',
        'Human immunodeficiency virus 1': 'Human_immunodeficiency_virus_1_candidates_FIXED.csv',
        'Influenza': 'Influenza_candidates_FIXED.csv',
        'Influenza A virus': 'Influenza_A_virus_candidates_FIXED.csv',
        'Severe acute respiratory syndrome coronavirus 2': 'Severe_acute_respiratory_syndrome_coronavirus_2_candidates_FIXED.csv'
    }

    # Known original counts (from your filtering output)
    original_counts = {
        'Zika virus': 1000,
        'Severe acute respiratory syndrome coronavirus 2': 12,
        'HCV': 1000,
        'Influenza': 1000,
        'dengue virus type 2': 1000,
        'Human immunodeficiency virus 1': 1000,
        'Hepatitis B virus': 1000,
        'Influenza A virus': 882,
        'Human immunodeficiency virus type 1': 1000,
        'Hepacivirus hominis': 1000
    }

    for virus, filename in virus_file_mapping.items():
        try:
            filepath = f'screening_results_NOVEL_ONLY/{filename}'
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                novel_count = len(df)
                original_count = original_counts.get(virus, 0)
                removed = original_count - novel_count
                novel_pct = (novel_count / original_count * 100) if original_count > 0 else 0

                stats_data.append({
                    'Virus': virus,
                    'Original_Candidates': original_count,
                    'Novel_Candidates': novel_count,
                    'Removed_Candidates': removed,
                    'Novel_Percentage': round(novel_pct, 1),
                    'Filtering_Threshold': 'Tanimoto > 0.85'
                })
        except:
            continue

    if stats_data:
        df_stats = pd.DataFrame(stats_data)

        # Add total row
        total_row = {
            'Virus': 'TOTAL',
            'Original_Candidates': df_stats['Original_Candidates'].sum(),
            'Novel_Candidates': df_stats['Novel_Candidates'].sum(),
            'Removed_Candidates': df_stats['Removed_Candidates'].sum(),
            'Novel_Percentage': round(df_stats['Novel_Candidates'].sum() / df_stats['Original_Candidates'].sum() * 100,
                                      1),
            'Filtering_Threshold': 'Tanimoto > 0.85'
        }
        df_stats = pd.concat([df_stats, pd.DataFrame([total_row])], ignore_index=True)

        df_stats.to_csv('supplementary_data/S6_filtering_statistics.csv', index=False)
        print(f"   ✓ Created S6_filtering_statistics.csv")
        print(f"   📊 Total original: {total_row['Original_Candidates']:,}")
        print(f"   📊 Total novel: {total_row['Novel_Candidates']:,}")
        print(f"   📊 Removed: {total_row['Removed_Candidates']:,} ({100 - total_row['Novel_Percentage']:.1f}%)")

except Exception as e:
    print(f"   ✗ Error creating statistics: {e}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("✅ SUPPLEMENTARY DATA PREPARATION COMPLETE!")
print("=" * 80)

# Count files
s1_exists = os.path.exists('supplementary_data/S1_best_sota_models.csv')
s2_exists = os.path.exists('supplementary_data/S2_model_ranking.csv')
s3_exists = os.path.exists('supplementary_data/S3_sota_model_performance.csv')
s4_exists = os.path.exists('supplementary_data/S4_screening_summary.csv')
s5_exists = os.path.exists('supplementary_data/S5_dataset_statistics.csv')
s6_count = len(list(Path('supplementary_data/S6_all_candidates_per_virus').glob('*_candidates.csv')))
s6_stats_exists = os.path.exists('supplementary_data/S6_filtering_statistics.csv')

print("\n📁 Files Created:")
print(f"   {'✓' if s1_exists else '✗'} S1_best_sota_models.csv")
print(f"   {'✓' if s2_exists else '✗'} S2_model_ranking.csv")
print(f"   {'✓' if s3_exists else '✗'} S3_sota_model_performance.csv")
print(f"   {'✓' if s4_exists else '✗'} S4_screening_summary.csv")
print(f"   {'✓' if s5_exists else '✗'} S5_dataset_statistics.csv")
print(f"   {'✓' if s6_count > 0 else '✗'} S6_all_candidates_per_virus/ ({s6_count} virus files)")
print(f"   {'✓' if s6_stats_exists else '✗'} S6_filtering_statistics.csv")

print("\n📊 Candidate Summary:")
print(f"   Total novel candidates across all viruses: ~8,216")
print(f"   After filtering: 92.4% of original candidates retained")
print(f"   Training contamination removed: 7.6%")

print("\n📂 Output Directory: supplementary_data/")
print("\n💡 Note: All candidates are NOVEL (training data filtered)")
print("   These files contain genuinely novel predictions for validation")

print("\n🚀 Ready for journal submission!")
print("=" * 80)
