import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# -------------------------------
# Configuration
# -------------------------------
folder_path = "./"
save_dir = os.path.join(folder_path, "insulation_score_correlation/")
os.makedirs(save_dir, exist_ok=True)

cell_types = ["IMR90"]
model_names = ["ChromNet"]

# Load chromosome data
chr_data = pd.read_csv(os.path.join(folder_path, "hg38.chrom.sizes.txt"), sep='\t', header=None)
chr_data.columns = ["chrom_name", "chrom_length"]
chr_names = chr_data["chrom_name"].values[19:22]
chr_lens = chr_data["chrom_length"].values[19:22]

# -------------------------------
# Functions to calculate correlations
# -------------------------------

def insulation_correlation(preds, targets, method='pearson'):
    """
    Compute the correlation score between predicted and target insulation scores.

    Parameters
    ----------
    preds : str
        Path to the predicted insulation scores.
    targets : str
        Path to the target insulation scores.
    method : str, optional
        The method to compute correlation, either 'pearson' or 'spearman'. Default is 'pearson'.

    Returns
    -------
    results : list
        List of correlation scores for each chromosome.
    """
    scores = []
    pred_data = np.load(preds)
    target_data = np.load(targets)
    num_data = pred_data.shape[0]
    
    for idx in range(num_data):
        pred_insu = pred_data[idx]
        label_insu = target_data[idx]
        
        if method == 'pearson':
            metric, _ = pearsonr(pred_insu, label_insu)
        elif method == 'spearman':
            metric, _ = spearmanr(pred_insu, label_insu)
        else:
            raise ValueError("Method should be either 'pearson' or 'spearman'")
        
        scores.append(metric)
    
    return scores

# -------------------------------
# Main loop to process each cell type and model
# -------------------------------

for cell_type in cell_types:
    pearson_results_dict = {"methods": []}
    spearman_results_dict = {"methods": []}
    
    # Initialize dictionary keys for chromosomes
    for chr_name in chr_names:
        pearson_results_dict[chr_name] = []
        spearman_results_dict[chr_name] = []
    
    print(f"------------------------------- {cell_type} -------------------------------")
    
    for model_name in model_names:
        pearson_results_dict["methods"].append(model_name)
        spearman_results_dict["methods"].append(model_name)
        
        for chr_name in chr_names:
            # Define paths
            data_orign_dir = os.path.join(folder_path, f"insulation_hicplotter_cal_result/{cell_type}/")
            data_dir = os.path.join(folder_path, f"insulation_hicplotter_cal_result/{cell_type}/")
            
            targets = os.path.join(data_orign_dir, f"orign_all_insulation_score_{chr_name}.npy")
            preds = os.path.join(data_dir, f"{model_name}_all_insulation_score_{chr_name}.npy")
            
            # Calculate Pearson correlation
            p_results = insulation_correlation(preds, targets, method='pearson')
            p_filtered_results = np.array(p_results)[~np.isnan(p_results)]
            pearson_results_dict[chr_name].append(np.mean(p_filtered_results))
            
            # Calculate Spearman correlation
            s_results = insulation_correlation(preds, targets, method='spearman')
            s_filtered_results = np.array(s_results)[~np.isnan(s_results)]
            spearman_results_dict[chr_name].append(np.mean(s_filtered_results))
    
    # Save results for Pearson correlation
    pearson_results_df = pd.DataFrame(pearson_results_dict)
    pearson_results_df["avg"] = pearson_results_df[chr_names].mean(axis=1)
    print("Insulation score Pearson correlation result:")
    print(pearson_results_df)
    pearson_results_df.to_csv(os.path.join(save_dir, f"pearson_results_{cell_type}.csv"), index=False)
    
    # Save results for Spearman correlation
    spearman_results_df = pd.DataFrame(spearman_results_dict)
    spearman_results_df["avg"] = spearman_results_df[chr_names].mean(axis=1)
    print("Insulation score Spearman correlation result:")
    print(spearman_results_df)
    spearman_results_df.to_csv(os.path.join(save_dir, f"spearman_results_{cell_type}.csv"), index=False)
