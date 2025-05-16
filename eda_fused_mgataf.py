import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
gdsc_df = pd.read_csv("dataset/GDSC_SMILES_merged.csv", index_col=0)
ccl_df = pd.read_csv("dataset/PANCANCER_Genetic_feature.csv")
gsva_df = pd.read_csv("dataset/ccle_gsva_scores.csv", index_col=0)

# Clean
gdsc_df["CLEAN_CELL_LINE"] = gdsc_df["CELL_LINE_NAME"].str.upper().str.replace(r'\W+', '', regex=True)

# ---- 1. Drug & Cell Line Overview ----
print("Unique drugs:", gdsc_df["DRUG_ID"].nunique())
print("Unique cell lines:", gdsc_df["CLEAN_CELL_LINE"].nunique())

# ---- 2. IC50 Value Distribution ----
plt.figure(figsize=(8, 4))
sns.histplot(gdsc_df["LN_IC50"], bins=50, kde=True, color='salmon')
plt.title("Distribution of LN_IC50")
plt.xlabel("LN_IC50")
plt.tight_layout()
plt.savefig("plots/ic50_distribution.pdf")
plt.close()

# ---- 3. Count of samples per drug ----
# Map ID to name
drug_id_to_name = gdsc_df.drop_duplicates("DRUG_ID")[["DRUG_ID", "DRUG_NAME"]].set_index("DRUG_ID")["DRUG_NAME"]
drug_counts = gdsc_df["DRUG_ID"].value_counts()

# Create new Series with names instead of IDs
drug_counts_named = drug_counts.rename(index=drug_id_to_name)

# Plot
plt.figure(figsize=(12, 4))
drug_counts_named.head(30).plot(kind="bar", color='teal')
plt.title("Top 30 Drugs by Frequency")
plt.ylabel("Sample Count")
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.tight_layout()
plt.savefig("plots/top_drugs_named.pdf")
plt.close()


# ---- 4. Count of samples per cell line ----
cell_counts = gdsc_df["CLEAN_CELL_LINE"].value_counts()
plt.figure(figsize=(10, 3))
cell_counts.head(30).plot(kind="bar", color='orchid')
plt.title("Top 30 Cell Lines by Frequency")
plt.ylabel("Sample Count")
plt.tight_layout()
plt.savefig("plots/top_cell_lines.pdf")
plt.close()

# ---- 5. GSVA: Pathway-wise variance ----
gsva_df.columns = gsva_df.columns.str.split("_").str[0].str.upper()
gsva_df = gsva_df.loc[:, ~gsva_df.columns.duplicated()]
gsva_matrix = gsva_df.T  # shape: cell lines x pathways
pathway_variance = gsva_matrix.var().sort_values(ascending=False)

plt.figure(figsize=(14, 6))
# Clean pathway names
cleaned_names = pathway_variance.head(30).index.str.replace(r"^KEGG_MEDICUS_", "", regex=True)
# Plot with cleaned names
ax = pathway_variance.head(30).plot(kind="bar", color='skyblue')
ax.set_xticklabels(cleaned_names, rotation=45, ha='right', fontsize=9)
ax.set_xticklabels([label.get_text()[:15] + '...' for label in ax.get_xticklabels()])
plt.title("Top 30 Pathways by Variance (GSVA)")
plt.ylabel("Variance")
plt.tight_layout()
plt.savefig("plots/gsva_variance.pdf")
plt.close()


# ---- 6. Mutation/CNV heatmap preview ----
pivot = ccl_df.pivot_table(index="cell_line_name", columns="genetic_feature", values="is_mutated", fill_value=0)
plt.figure(figsize=(12, 5))
sns.heatmap(pivot.sample(n=50, random_state=42), cmap="viridis", cbar=False)
plt.title("Mutation/CNV Matrix (sampled)")
plt.tight_layout()
plt.savefig("plots/mut_cnv_heatmap.pdf")
plt.close()

# ---- 7. SMILES Length Distribution ----
gdsc_df["SMILES_LEN"] = gdsc_df["SMILES"].astype(str).apply(len)
plt.figure(figsize=(7, 4))
sns.histplot(gdsc_df["SMILES_LEN"], bins=50, color="coral")
plt.title("SMILES String Lengths")
plt.xlabel("Length")
plt.tight_layout()
plt.savefig("plots/smiles_length.pdf")
plt.close()

# ---- 8. Missing Data Check ----
print("\nMissing value report:")
print(gdsc_df.isna().mean() * 100)

# ---- 9. Missing Data Check ----
# Apply sigmoid-like normalization
def normalize_ic50(y):
    return 1 / (1 + np.exp(-0.1 * y))

gdsc_df["IC50_NORM"] = normalize_ic50(gdsc_df["LN_IC50"])

# Plot distributions
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(gdsc_df["LN_IC50"], bins=50, kde=True, color="coral")
plt.title("Original LN_IC50 Distribution")
plt.xlabel("LN_IC50")

plt.subplot(1, 2, 2)
sns.histplot(gdsc_df["IC50_NORM"], bins=50, kde=True, color="seagreen")
plt.title("Normalized IC50 (Sigmoid Transform)")
plt.xlabel("IC50_NORM")

plt.tight_layout()
plt.savefig("plots/ic50_normalized_vs_raw.pdf")
plt.show()