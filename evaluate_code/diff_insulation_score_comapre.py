import numpy as np
import os
import pandas as pd
from dtaidistance import dtw
import warnings
from scipy.stats import zscore

warnings.filterwarnings("ignore")

folder_path = "./"

def compute_dtw(preds1, preds2):
    scores = []
    num_data = preds1.shape[0]
    
    for idx in range(num_data):
        data1 = preds1[idx]
        data2 = preds2[idx]
        dtw_dist = dtw.distance(data1, data2)
        scores.append(dtw_dist)
    return scores

methods = ["ChromNet", "COrigami", "orign"]
cell_types = ["PBMC_103", "AML_018", "AML_472", "AML_546", "AML_270", "AML_629", "AML_168", "AML_027"]

chr_data = pd.read_csv(folder_path + "hg38.chrom.sizes.txt", sep='\t', header=None)
chr_data.columns = ["chrom_name", "chrom_length"]
all_chr_names = chr_data["chrom_name"].values
chr_names = all_chr_names[19:22]

save_dir = folder_path + "method_comparison_results_zscore/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 存储不同方法内部的细胞差异
method_differences = {method: {} for method in methods}

for method in methods:
    print(f"Processing method: {method}")
    
    for i, cell_type1 in enumerate(cell_types):
        for cell_type2 in cell_types[i+1:]:
            pair_key = f"{cell_type1}_vs_{cell_type2}"
            method_differences[method][pair_key] = []
            
            for chr_name in chr_names:
                data_dir1 = folder_path + f"insulation_hicplotter/{cell_type1}/"
                data_dir2 = folder_path + f"insulation_hicplotter/{cell_type2}/"
                
                file1 = data_dir1 + f"{method}_all_insulation_score_{chr_name}.npy"
                file2 = data_dir2 + f"{method}_all_insulation_score_{chr_name}.npy"
                
                try:
                    data1 = np.load(file1)
                    data2 = np.load(file2)
                    diff = zscore(data1) - zscore(data2)
                    method_differences[method][pair_key].append(diff)
                    
                    # 保存中间结果
                    np.save(save_dir + f"diff_{method}_{pair_key}_{chr_name}.npy", diff)
                    
                    # print(f"Saved diff for {method}, {pair_key}, {chr_name} with shape {diff.shape}")
                except Exception as e:
                    print(f"Error processing {pair_key} on {chr_name}: {e}")

dtw_results = {}

for pair_key in method_differences["ChromNet"].keys():
    dtw_results[pair_key] = []
    
    for chr_idx in range(len(chr_names)):
        # try:
        diffs = {method: method_differences[method][pair_key][chr_idx] for method in methods}

        dtw_dist_ChromNet = compute_dtw(diffs["orign"], diffs["ChromNet"])
        dtw_dist_COrigami = compute_dtw(diffs["orign"], diffs["COrigami"])

        dtw_results[pair_key].append([dtw_dist_ChromNet,dtw_dist_COrigami])



