import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

labels = pd.read_csv("../data/stage_1_train_labels.csv")
detailed = pd.read_csv("../data/stage_1_detailed_class_info.csv")
data = labels.merge(detailed, on='patientId', how='left')

print("=== DATASET PROFILING ===")
print(f"Total images: {len(data)}")
print(f"Class distribution:\n{data['Target'].value_counts(normalize=True)}")
print(f"Ambiguous cases (No Lung Opacity/Not Normal): {(data['class'] == 'No Lung Opacity/Not Normal').sum()}")

# Plot class distribution
plt.figure(figsize=(8,5))
sns.countplot(x='Target', data=data)
plt.title("Class Distribution (0=Normal, 1=Disease)")
plt.savefig("../outputs/class_distribution.png")
plt.close()

# Patient leakage check
print(f"Unique patients: {data['patientId'].nunique()}")
print(f"Max images per patient: {data['patientId'].value_counts().max()} → leakage risk if not grouped!")

print("Profiling plots saved in outputs/. Deep analysis ready for report.")