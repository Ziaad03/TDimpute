import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Use low_memory=False to avoid dtype guessing issues
file_path = "RNA_DNA_combine.csv"


shuffle_cancer = pd.read_csv(file_path, delimiter=',', index_col=0, header=0)
print( ' data size:',shuffle_cancer.shape)
# Only load first 5 rows to inspect structure
print("🔍 Reading first few lines to understand structure...")
sample = pd.read_csv(file_path, nrows=3)
print(sample.head())
sample.info()
sample.head()

# # Split data sets into 3 parts : labels which is the first column and colummns that start with "?|", and the rest
# label_cols = [col for col in sample.columns if col == sample.columns[0]]
# dnamthylation_cols = [col for col in sample.columns if col.startswith("?|")]
# gene_expression_cols = [col for col in sample.columns if col not in label_cols + dnamthylation_cols]

# put the first 19028 in GE and the rest in DNA methylation
gene_expression_cols = sample.columns[:19027].tolist()  
dnamthylation_cols = sample.columns[19027:].tolist()

print(f"length of dna mthylation columns:{len(dnamthylation_cols)}")

# put the dna mthylation in a seprate csv file
dna_methylation_data = sample[dnamthylation_cols]
dna_methylation_data.to_csv("sample_DNA_methylation.csv", index=False)

# print(f"\nLabel columns: {len(label_cols)}")
# print (f"\nlength of dna mthylation columns:{len(dnamthylation_cols)}")
# print(f"\ngene expression cols: {len(gene_expression_cols)}")
""" print("\nColumns:", len(sample.columns))
print(sample.columns[:20])  # preview first 20 columns """


""" #Split DNA mthylation and Gene expression data 
cpg_cols = []
for col in sample.columns:


# write the 5 extracted rows back in csv file 
sample.to_csv("sample_RNA_DNA_combine.csv", index=False) """


""" # Efficiently load full data in chunks
print("\n📦 Counting rows (samples)...")
total_rows = sum(1 for _ in open(file_path)) - 1
print(f"Total samples (rows): {total_rows}")

# Load only header first to inspect feature names
df_head = pd.read_csv(file_path, nrows=1)
columns = df_head.columns.tolist()

# Identify column groups (rough guess)
rna_cols = [c for c in columns if "gene" in c.lower() or "exp" in c.lower()]
dna_cols = [c for c in columns if "meth" in c.lower() or "cg" in c.lower()]
label_cols = [c for c in columns if "cancer" in c.lower() or "type" in c.lower() or "id" in c.lower()]

print(f"\n🧬 Estimated gene expression features: {len(rna_cols)}")
print(f"🧫 Estimated DNA methylation features: {len(dna_cols)}")
print(f"🏷️ Estimated label/meta columns: {len(label_cols)}")

# If memory allows (<=16 GB RAM), load first 10,000 samples to explore
print("\n📥 Loading subset (10,000 rows) for analysis...")
subset = pd.read_csv(file_path, nrows=10000)

print("\n🔹 Subset shape:", subset.shape)
print(subset.info())

# --- Missing values ---
print("\n🕳️ Missing value stats:")
missing = subset.isnull().mean().sort_values(ascending=False)
print(missing.head(20))

plt.figure(figsize=(10,5))
sns.histplot(missing, bins=50)
plt.title("Distribution of Missing Value Fraction per Feature")
plt.xlabel("Fraction Missing")
plt.ylabel("Number of Features")
plt.show()

# --- Basic value distribution ---
numeric_cols = subset.select_dtypes(include=np.number).columns
sampled_cols = np.random.choice(numeric_cols, min(10, len(numeric_cols)), replace=False)

subset[sampled_cols].hist(figsize=(15,10), bins=30)
plt.suptitle("Value Distribution of Sampled Features")
plt.show()

# --- Correlation heatmap (small sample) ---
corr_sample = subset[sampled_cols].corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr_sample, cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap (Sampled Features)")
plt.show()
 """