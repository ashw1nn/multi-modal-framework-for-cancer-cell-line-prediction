import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv("results/mgataf_config_results.csv")
x_pos = np.arange(len(df))

# === Plot 1: PCC ===
plt.figure(figsize=(8, 5))
plt.errorbar(
    y=df["PCC"], x=x_pos, yerr=df["PCC_SD"],
    fmt='o', color='blue', ecolor='dodgerblue', capsize=6,
    elinewidth=4, marker='s', markersize=6
)
plt.xticks(x_pos, df["Configuration"], rotation=45, ha='right', fontsize=9)
plt.title("Model Comparison: PCC (± SD)")
plt.ylabel("PCC")
plt.xlabel("Configuration")
plt.savefig("plots/results_pcc_comparision.pdf", dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()


# === Plot 2: RMSE ===
plt.figure(figsize=(8, 5))
plt.errorbar(
    y=df["RMSE"], x=x_pos, yerr=df["RMSE_SD"],
    fmt='o', color='darkred', ecolor='salmon', capsize=6,
    elinewidth=4, marker='s', markersize=6
)
plt.xticks(x_pos, df["Configuration"], rotation=45, ha='right', fontsize=9)
plt.title("Model Comparison: RMSE (± SD)")
plt.ylabel("RMSE")
plt.xlabel("Configuration")
plt.savefig("plots/results_rmse_comparision.pdf", dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()
