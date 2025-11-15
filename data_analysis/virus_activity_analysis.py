import pandas as pd
import matplotlib.pyplot as plt

# Read the dataset - it's tab-separated, not comma-separated
df = pd.read_csv('../data/virus_activity/filtered/main_dataset.csv',
                 sep='\t')

# Display basic information
print("=" * 80)
print("DATASET OVERVIEW")
print("=" * 80)
print(f"Total compounds: {len(df)}")
print(f"Number of viruses: {df['virus_name'].nunique()}")
print(f"\nViruses in dataset: {sorted(df['virus_name'].unique())}")
print(f"\nActivity types: {df['activity_type'].unique()}")
print(f"Activity units: {df['activity_unit'].unique()}")

# Define activity threshold
# Typically, compounds with IC50/EC50 < 10,000 nM (10 μM) are considered active
ACTIVITY_THRESHOLD = 10000  # nM


# For Ki values, same threshold applies
# For inhibition percentage, we'll use > 50% as active

def classify_activity(row):
    """
    Classify compound as active or inactive based on activity value and type
    """
    activity_type = row['activity_type']
    activity_value = row['activity_value']

    # Handle different activity types
    if activity_type in ['IC50', 'EC50', 'Ki']:
        # Lower values = more active (more potent)
        return 'Active' if activity_value < ACTIVITY_THRESHOLD else 'Inactive'
    elif 'Inhibition' in activity_type:
        # Higher percentage = more active
        return 'Active' if activity_value > 50 else 'Inactive'
    else:
        # Default: use threshold
        return 'Active' if activity_value < ACTIVITY_THRESHOLD else 'Inactive'


# Classify all compounds
df['activity_class'] = df.apply(classify_activity, axis=1)

# Count active and inactive per virus
print("\n" + "=" * 80)
print("ACTIVE vs INACTIVE COMPOUNDS PER VIRUS")
print("=" * 80)
print(f"(Threshold: IC50/EC50/Ki < {ACTIVITY_THRESHOLD} nM = Active)")
print(f"(Threshold: Inhibition > 50% = Active)")
print("=" * 80)

virus_activity = df.groupby(['virus_name', 'activity_class']).size().unstack(fill_value=0)
virus_activity['Total'] = virus_activity.sum(axis=1)
virus_activity['Active %'] = (virus_activity['Active'] / virus_activity['Total'] * 100).round(2)
virus_activity['Inactive %'] = (virus_activity['Inactive'] / virus_activity['Total'] * 100).round(2)

# Sort by total count
virus_activity = virus_activity.sort_values('Total', ascending=False)

print(virus_activity)

# Detailed breakdown
print("\n" + "=" * 80)
print("DETAILED BREAKDOWN BY VIRUS")
print("=" * 80)
for virus in virus_activity.index:
    active_count = virus_activity.loc[virus, 'Active']
    inactive_count = virus_activity.loc[virus, 'Inactive']
    total = virus_activity.loc[virus, 'Total']
    active_pct = virus_activity.loc[virus, 'Active %']

    print(f"\n{virus}:")
    print(f"  Active:   {active_count:>8,} compounds ({active_pct:>6.2f}%)")
    print(f"  Inactive: {inactive_count:>8,} compounds ({100 - active_pct:>6.2f}%)")
    print(f"  Total:    {total:>8,} compounds")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Stacked bar chart
ax1 = axes[0, 0]
virus_activity[['Active', 'Inactive']].plot(kind='barh', stacked=True, ax=ax1,
                                            color=['#2ecc71', '#e74c3c'])
ax1.set_xlabel('Number of Compounds', fontsize=12, fontweight='bold')
ax1.set_ylabel('Virus', fontsize=12, fontweight='bold')
ax1.set_title('Active vs Inactive Compounds per Virus (Stacked)',
              fontsize=14, fontweight='bold', pad=20)
ax1.legend(title='Activity Class', fontsize=10)
ax1.grid(axis='x', alpha=0.3)

# 2. Grouped bar chart
ax2 = axes[0, 1]
virus_activity[['Active', 'Inactive']].plot(kind='barh', ax=ax2,
                                            color=['#2ecc71', '#e74c3c'])
ax2.set_xlabel('Number of Compounds', fontsize=12, fontweight='bold')
ax2.set_ylabel('Virus', fontsize=12, fontweight='bold')
ax2.set_title('Active vs Inactive Compounds per Virus (Grouped)',
              fontsize=14, fontweight='bold', pad=20)
ax2.legend(title='Activity Class', fontsize=10)
ax2.grid(axis='x', alpha=0.3)

# 3. Percentage stacked bar chart
ax3 = axes[1, 0]
virus_activity_pct = virus_activity[['Active', 'Inactive']].div(virus_activity['Total'], axis=0) * 100
virus_activity_pct.plot(kind='barh', stacked=True, ax=ax3,
                        color=['#2ecc71', '#e74c3c'])
ax3.set_xlabel('Percentage (%)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Virus', fontsize=12, fontweight='bold')
ax3.set_title('Percentage of Active vs Inactive Compounds per Virus',
              fontsize=14, fontweight='bold', pad=20)
ax3.legend(title='Activity Class', fontsize=10)
ax3.grid(axis='x', alpha=0.3)

# Add percentage labels
for i, virus in enumerate(virus_activity_pct.index):
    active_pct = virus_activity_pct.loc[virus, 'Active']
    inactive_pct = virus_activity_pct.loc[virus, 'Inactive']
    ax3.text(active_pct / 2, i, f'{active_pct:.1f}%',
             ha='center', va='center', fontweight='bold', color='white')
    ax3.text(active_pct + inactive_pct / 2, i, f'{inactive_pct:.1f}%',
             ha='center', va='center', fontweight='bold', color='white')

# 4. Pie chart for overall distribution
ax4 = axes[1, 1]
total_active = virus_activity['Active'].sum()
total_inactive = virus_activity['Inactive'].sum()
colors = ['#2ecc71', '#e74c3c']
ax4.pie([total_active, total_inactive], labels=['Active', 'Inactive'],
        autopct='%1.1f%%', startangle=90, colors=colors, textprops={'fontsize': 12})
ax4.set_title(f'Overall Distribution\n(Total: {total_active + total_inactive:,} compounds)',
              fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('virus_activity_analysis/virus_activity_distribution.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Visualization saved to: virus_activity_distribution.png")

# Export summary to CSV
virus_activity.to_csv('virus_activity_analysis/virus_activity_summary.csv')
print(f"✓ Summary table saved to: virus_activity_summary.csv")

# Additional analysis: Activity value distribution per virus
print("\n" + "=" * 80)
print("ACTIVITY VALUE STATISTICS BY VIRUS")
print("=" * 80)

for virus in sorted(df['virus_name'].unique()):
    virus_data = df[df['virus_name'] == virus]
    print(f"\n{virus}:")
    print(
        f"  Activity value range: {virus_data['activity_value'].min():.2f} - {virus_data['activity_value'].max():.2f}")
    print(f"  Mean activity value: {virus_data['activity_value'].mean():.2f}")
    print(f"  Median activity value: {virus_data['activity_value'].median():.2f}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
