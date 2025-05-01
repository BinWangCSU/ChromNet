import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from dtaidistance import dtw
import warnings
warnings.filterwarnings("ignore")

folder_path = "./"
data_path=folder_path + "method_comparison_results/"
save_dir = folder_path + "plot_cell_diff_results/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Define methods and cell types
methods = ["ChromNet", "C.Origami", "orign"]
cell_types = ["PBMC_103", "AML_018", "AML_472", "AML_546", "AML_270", "AML_629", "AML_168", "AML_027"]
chr_data = pd.read_csv(folder_path + "hg38.chrom.sizes.txt", sep='\t', header=None)
chr_data.columns = ["chrom_name", "chrom_length"]
all_chr_names = chr_data["chrom_name"].values
chr_names = all_chr_names[19:22]  # chromosomes 20-22

# Extract only PBMC vs other cells
pbmc_comparisons = []
for cell_type in cell_types[1:]:  # Skip PBMC itself
    pbmc_comparisons.append(f"PBMC_103_vs_{cell_type}")

# Function to compute DTW distances (maintain from original code)
def compute_dtw(preds1, preds2):
    scores = []
    num_data = preds1.shape[0]
    for idx in range(num_data):
        data1 = preds1[idx]
        data2 = preds2[idx]
        dtw_dist = dtw.distance(data1, data2)
        scores.append(dtw_dist)
    return scores

# Integrated approach: Load and combine data from all chromosomes
integrated_dtw_results = {}

for pair_key in pbmc_comparisons:
    cell_comparison = pair_key.split("_vs_")[1]
    integrated_dtw_results[cell_comparison] = {"ChromNet": [], "C.Origami": []}
    
    for chr_name in chr_names:
        try:
            # Load the previously saved difference files
            diff_orign = np.load(data_path + f"diff_orign_{pair_key}_{chr_name}.npy")
            diff_ChromNet = np.load(data_path + f"diff_ChromNet_{pair_key}_{chr_name}.npy")
            diff_COrigami = np.load(data_path + f"diff_COrigami_{pair_key}_{chr_name}.npy")
            
            # Compute distances
            dtw_dist_ChromNet = compute_dtw(diff_orign, diff_ChromNet)
            dtw_dist_COrigami = compute_dtw(diff_orign, diff_COrigami)
            
            # Add to integrated results
            integrated_dtw_results[cell_comparison]["ChromNet"].extend(dtw_dist_ChromNet)
            integrated_dtw_results[cell_comparison]["C.Origami"].extend(dtw_dist_COrigami)
            
        except Exception as e:
            print(f"Error processing {pair_key} on {chr_name}: {e}")

# Prepare data for visualization
def prepare_integrated_data(integrated_results, metric_name):
    data_list = []
    
    for cell_type, method_data in integrated_results.items():
        for method, scores in method_data.items():
            for score in scores:
                data_list.append({
                    "Cell Type": cell_type,
                    "Method": method,
                    f"{metric_name} Distance": score
                })
    
    return pd.DataFrame(data_list)

# Create integrated dataframes for visualization
integrated_dtw_df = prepare_integrated_data(integrated_dtw_results, "DTW")

# Set Nature Biotechnology compatible style
plt.style.use('default')
sns.set(style="ticks", font_scale=1.2)
colors = ["#2171b5", "#6baed6"]  # Blue color palette

# 1. Create integrated violin plots for DTW distances
plt.figure(figsize=(10, 6))
ax = sns.violinplot(
    x="Cell Type", 
    y="DTW Distance", 
    hue="Method",
    data=integrated_dtw_df,
    palette=colors,
    split=True,
    inner="quartile",
    cut=0
)

# Add swarm plot for individual data points
sns.swarmplot(
    x="Cell Type", 
    y="DTW Distance", 
    hue="Method",
    data=integrated_dtw_df,
    palette=["#08306b", "#4292c6"],
    alpha=0.5,
    size=3,
    dodge=True,
    ax=ax
)

plt.title("DTW Distance: PBMC vs AML Cell Types", fontsize=14, fontweight='bold')
plt.xlabel("Cell Type", fontsize=12, fontweight='bold')
plt.ylabel("DTW Distance", fontsize=12, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles[:2], labels[:2], title="Method", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(save_dir + "Integrated_DTW_PBMC_comparison_NatureBiotech.png", dpi=300, bbox_inches='tight')
plt.close()

# 3. Create integrated bar plot comparing method performance
# Calculate mean values for each cell type and method
dtw_means = integrated_dtw_df.groupby(['Cell Type', 'Method'])['DTW Distance'].mean().reset_index()

# Pivot the data for easier plotting
dtw_pivot = dtw_means.pivot(index='Cell Type', columns='Method', values='DTW Distance')

# Calculate improvement ratios
dtw_pivot['Improvement Ratio'] = dtw_pivot['ChromNet'] / dtw_pivot['C.Origami']

# Create figure for DTW comparison
plt.figure(figsize=(10, 6))
ax = dtw_pivot[['ChromNet', 'C.Origami']].plot(kind='bar', color=colors)
plt.title('Mean DTW Distance: PBMC vs AML Cell Types', fontsize=14, fontweight='bold')
plt.ylabel('DTW Distance', fontsize=12, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xlabel('Cell Type', fontsize=12, fontweight='bold')

# Add improvement ratio as text
for i, cell in enumerate(dtw_pivot.index):
    ratio = dtw_pivot.loc[cell, 'Improvement Ratio']
    plt.text(i, max(dtw_pivot.loc[cell, 'ChromNet'], dtw_pivot.loc[cell, 'C.Origami']) * 1.05, 
             f'Ratio: {ratio:.2f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(save_dir + "Integrated_DTW_Method_Comparison_NatureBiotech.png", dpi=300, bbox_inches='tight')
plt.close()

# 4. Create a heatmap with integrated results
# Prepare data for heatmap
heatmap_data = pd.DataFrame(index=sorted(integrated_dtw_results.keys()))
heatmap_data['DTW ChromNet'] = [np.mean(integrated_dtw_results[cell]['ChromNet']) for cell in heatmap_data.index]
heatmap_data['DTW C.Origami'] = [np.mean(integrated_dtw_results[cell]['C.Origami']) for cell in heatmap_data.index]
heatmap_data['DTW Ratio (ChromNet/C.Origami)'] = heatmap_data['DTW ChromNet'] / heatmap_data['DTW C.Origami']

# Create heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    heatmap_data, 
    annot=True, 
    fmt=".3f", 
    cmap="YlGnBu", 
    linewidths=0.5,
    cbar_kws={"label": "Value"}
)
plt.title("Integrated DTW Comparison of ChromNet and C.Origami vs Original Data\nPBMC vs AML Cell Types (chr20-22)", 
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(save_dir + "Integrated_DTW_heatmap_NatureBiotech.png", dpi=300, bbox_inches='tight')
plt.close()

# 5. Statistical analysis and comparison
# Save the statistical summary to CSV
statistical_summary = pd.DataFrame({
    'Cell Type': sorted(integrated_dtw_results.keys()),
    'DTW ChromNet Mean': [np.mean(integrated_dtw_results[cell]['ChromNet']) for cell in sorted(integrated_dtw_results.keys())],
    'DTW ChromNet Std': [np.std(integrated_dtw_results[cell]['ChromNet']) for cell in sorted(integrated_dtw_results.keys())],
    'DTW C.Origami Mean': [np.mean(integrated_dtw_results[cell]['C.Origami']) for cell in sorted(integrated_dtw_results.keys())],
    'DTW C.Origami Std': [np.std(integrated_dtw_results[cell]['C.Origami']) for cell in sorted(integrated_dtw_results.keys())]
})

statistical_summary['DTW Improvement (%)'] = (statistical_summary['DTW ChromNet Mean'] - statistical_summary['DTW C.Origami Mean']) / statistical_summary['DTW ChromNet Mean'] * 100

statistical_summary.to_csv(save_dir + "Integrated_DTW_PBMC_comparison_statistics.csv", index=False)

print("Integrated DTW visualization across chromosomes 20-22 completed. Results saved to:", save_dir)