import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Use low_memory=False to avoid dtype guessing issues
file_path = "RNA_DNA_combine.csv"

# Only load first 5 rows to inspect structure
print("🔍 Reading first few lines to understand structure...")
sample = pd.read_csv(file_path, nrows=5)
print(sample.head())
print("\nColumns:", len(sample.columns))
print(sample.columns[:20])  # preview first 20 columns

# Efficiently load full data in chunks
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
