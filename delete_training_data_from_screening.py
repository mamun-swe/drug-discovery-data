"""
Filter Screening Results
Just keep what's not in training - nothing fancy
"""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit import DataStructs
from tqdm import tqdm
import os
import warnings

morgan_gen = GetMorganGenerator(
    radius=2,
    fpSize=2048,
    includeChirality=False,
    useBondTypes=True,
    countSimulation=False
)
warnings.filterwarnings('ignore')

print("=" * 80)
print("🔬 ULTRA SIMPLE FILTERING")
print("=" * 80)

# Settings
OUTPUT_DIR = 'screening_results_NOVEL_ONLY'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load training
print("\n📊 Loading training data...")
train_df = pd.read_csv('../data/virus_activity/cleaned_balanced_top10.csv', sep='\t')
print(f"   Loaded: {len(train_df):,} rows")

# Build training fingerprints (try multiple methods)
print("\n🧬 Processing training compounds...")
train_fps = []
train_inchikeys = []

for idx in tqdm(range(len(train_df)), desc="Training"):
    smiles_raw = train_df['smiles'].iloc[idx]

    # Skip bad values
    if pd.isna(smiles_raw) or not isinstance(smiles_raw, str):
        continue

    # Try parsing multiple ways
    mol = None

    # Method 1: As-is
    try:
        mol = Chem.MolFromSmiles(smiles_raw)
    except:
        pass

    # Method 2: Remove CXSMILES
    if mol is None and '|' in smiles_raw:
        try:
            clean = smiles_raw.split('|')[0].strip()
            mol = Chem.MolFromSmiles(clean)
        except:
            pass

    # Method 3: Remove CXSMILES and spaces
    if mol is None and '|' in smiles_raw:
        try:
            clean = smiles_raw.split('|')[0].strip().replace(' ', '')
            mol = Chem.MolFromSmiles(clean)
        except:
            pass

    if mol is not None:
        fp = morgan_gen.GetFingerprint(mol)
        inchi = Chem.MolToInchiKey(mol)
        train_fps.append(fp)
        train_inchikeys.append(inchi)

print(f"\n   ✅ Processed: {len(train_fps):,} valid training compounds")

if len(train_fps) == 0:
    print("\n   ❌ ERROR: No valid training compounds!")
    print("   Run: python debug_smiles.py")
    print("   And send me the output!")
    exit(1)

train_inchikey_set = set(train_inchikeys)

# Find screening files
print("\n📁 Finding screening files...")
screening_dir = 'screening_results'

if not os.path.exists(screening_dir):
    print(f"   ❌ Directory not found: {screening_dir}")
    print("   Where are your screening result files?")
    exit(1)

files = [f for f in os.listdir(screening_dir) if f.endswith('.csv')]
print(f"   Found {len(files)} files")

# Process each file
for filename in files:
    print(f"\n{'─' * 80}")
    print(f"Processing: {filename}")

    filepath = os.path.join(screening_dir, filename)
    df = pd.read_csv(filepath)
    print(f"   Loaded: {len(df):,} candidates")

    if 'smiles' not in df.columns:
        print(f"   ❌ No smiles column!")
        continue

    # Check each candidate
    keep_rows = []

    for idx in tqdm(range(len(df)), desc="   Filtering", leave=False):
        row = df.iloc[idx]
        smiles_raw = row['smiles']

        if pd.isna(smiles_raw) or not isinstance(smiles_raw, str):
            continue

        # Parse molecule (try multiple methods like training)
        mol = None
        try:
            mol = Chem.MolFromSmiles(smiles_raw)
        except:
            pass

        if mol is None and '|' in smiles_raw:
            try:
                clean = smiles_raw.split('|')[0].strip()
                mol = Chem.MolFromSmiles(clean)
            except:
                pass

        if mol is None:
            continue

        # Check if in training
        inchi = Chem.MolToInchiKey(mol)

        if inchi in train_inchikey_set:
            # Exact match - skip
            continue

        # Check similarity
        fp = morgan_gen.GetFingerprint(mol)
        max_sim = 0.0

        for train_fp in train_fps:
            sim = DataStructs.TanimotoSimilarity(fp, train_fp)
            if sim > max_sim:
                max_sim = sim

        # Keep if similarity < 0.85
        if max_sim < 0.85:
            keep_rows.append(idx)

    # Save
    if len(keep_rows) > 0:
        novel_df = df.iloc[keep_rows]
        output_path = os.path.join(OUTPUT_DIR, filename)
        novel_df.to_csv(output_path, index=False)

        removed = len(df) - len(keep_rows)
        print(f"   ✅ Kept: {len(keep_rows):,} ({len(keep_rows) / len(df) * 100:.1f}%)")
        print(f"   ❌ Removed: {removed:,} ({removed / len(df) * 100:.1f}%)")
        print(f"   💾 Saved: {output_path}")
    else:
        print(f"   ⚠️  No novel candidates!")

print("\n" + "=" * 80)
print("✅ DONE!")
print("=" * 80)
print(f"\nCheck: {OUTPUT_DIR}/")
