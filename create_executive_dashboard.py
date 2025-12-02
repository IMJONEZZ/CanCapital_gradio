#!/usr/bin/env python3
"""
Create executive dashboard visualization
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

# Load data
df = pd.read_csv("executive_predictions_summary.csv")

# Create dashboard
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Collection Case Predictions - Executive Dashboard', fontsize=20, fontweight='bold')

# 1. Distribution of Predictions
ax1 = axes[0, 0]
df['predicted_collection_days'].hist(bins=30, ax=ax1, color='steelblue', edgecolor='black')
ax1.axvline(df['predicted_collection_days'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["predicted_collection_days"].mean():.2f}')
ax1.set_xlabel('Predicted Days', fontsize=12)
ax1.set_ylabel('Number of Cases', fontsize=12)
ax1.set_title('Distribution of Predicted Collection Days', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Category Breakdown
ax2 = axes[0, 1]
category_counts = df['prediction_category'].value_counts()
colors = ['#2ecc71', '#f39c12', '#e74c3c']
wedges, texts, autotexts = ax2.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', 
                                     colors=colors, startangle=90)
ax2.set_title('Cases by Resolution Speed', fontsize=14, fontweight='bold')
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

# 3. Top 10 Longest Predicted Cases
ax3 = axes[0, 2]
top_10 = df.nlargest(10, 'predicted_collection_days')[['contract_number', 'predicted_collection_days']]
ax3.barh(range(len(top_10)), top_10['predicted_collection_days'], color='coral')
ax3.set_yticks(range(len(top_10)))
ax3.set_yticklabels(top_10['contract_number'].values, fontsize=9)
ax3.set_xlabel('Predicted Days', fontsize=12)
ax3.set_title('Top 10 Longest Predicted Cases', fontsize=14, fontweight='bold')
ax3.invert_yaxis()
ax3.grid(True, alpha=0.3, axis='x')

# 4. Actual vs Predicted (if actual exists)
ax4 = axes[1, 0]
if 'collection_case_days' in df.columns:
    df_with_actual = df.dropna(subset=['collection_case_days'])
    if len(df_with_actual) > 0:
        ax4.scatter(df_with_actual['collection_case_days'], 
                   df_with_actual['predicted_collection_days'], 
                   alpha=0.5, s=30, color='steelblue')
        max_val = max(df_with_actual['collection_case_days'].max(), 
                     df_with_actual['predicted_collection_days'].max())
        ax4.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        ax4.set_xlabel('Actual Days', fontsize=12)
        ax4.set_ylabel('Predicted Days', fontsize=12)
        ax4.set_title(f'Actual vs Predicted ({len(df_with_actual)} cases)', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No actual data available', ha='center', va='center', fontsize=14)
        ax4.set_title('Actual vs Predicted', fontsize=14, fontweight='bold')
else:
    ax4.text(0.5, 0.5, 'No actual data available', ha='center', va='center', fontsize=14)
    ax4.set_title('Actual vs Predicted', fontsize=14, fontweight='bold')

# 5. Summary Statistics Table
ax5 = axes[1, 1]
ax5.axis('off')
stats_data = [
    ['Total Cases', f"{len(df):,}"],
    ['Mean Prediction', f"{df['predicted_collection_days'].mean():.2f} days"],
    ['Median Prediction', f"{df['predicted_collection_days'].median():.2f} days"],
    ['Std Deviation', f"{df['predicted_collection_days'].std():.2f} days"],
    ['', ''],
    ['Quick (< 2.5 days)', f"{(df['prediction_category'] == 'Quick Resolution (< 2.5 days)').sum():,} ({(df['prediction_category'] == 'Quick Resolution (< 2.5 days)').sum()/len(df)*100:.1f}%)"],
    ['Medium (2.5-4 days)', f"{(df['prediction_category'] == 'Medium (2.5-4 days)').sum():,} ({(df['prediction_category'] == 'Medium (2.5-4 days)').sum()/len(df)*100:.1f}%)"],
    ['Extended (> 4 days)', f"{(df['prediction_category'] == 'Extended (> 4 days)').sum():,} ({(df['prediction_category'] == 'Extended (> 4 days)').sum()/len(df)*100:.1f}%)"],
]
table = ax5.table(cellText=stats_data, cellLoc='left', loc='center',
                 colWidths=[0.5, 0.5])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)
for i in range(len(stats_data)):
    if i == 4:  # Empty row
        continue
    cell = table[(i, 0)]
    cell.set_facecolor('#f0f0f0')
    cell.set_text_props(weight='bold')
ax5.set_title('Summary Statistics', fontsize=14, fontweight='bold', pad=20)

# 6. Collection Status Breakdown
ax6 = axes[1, 2]
if 'collection_status' in df.columns:
    status_counts = df['collection_status'].value_counts().head(8)
    ax6.barh(range(len(status_counts)), status_counts.values, color='teal')
    ax6.set_yticks(range(len(status_counts)))
    ax6.set_yticklabels(status_counts.index, fontsize=9)
    ax6.set_xlabel('Number of Cases', fontsize=12)
    ax6.set_title('Cases by Collection Status', fontsize=14, fontweight='bold')
    ax6.invert_yaxis()
    ax6.grid(True, alpha=0.3, axis='x')
else:
    ax6.text(0.5, 0.5, 'Status data not available', ha='center', va='center', fontsize=14)
    ax6.set_title('Cases by Collection Status', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('executive_predictions_dashboard.png', dpi=300, bbox_inches='tight')
print("✅ Dashboard saved to: executive_predictions_dashboard.png")
plt.close()

print("\n" + "="*60)
print("EXECUTIVE PRESENTATION PACKAGE READY")
print("="*60)
print("\nFiles created:")
print("1. executive_predictions_report.csv (Full dataset with predictions)")
print("2. executive_predictions_summary.csv (Key columns only)")
print("3. executive_predictions_dashboard.png (Visual dashboard)")
print("\nReady to present to executives!")
