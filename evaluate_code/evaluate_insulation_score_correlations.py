import numpy as np
import os
import pandas as pd
import argparse
from scipy.stats import pearsonr, spearmanr
from math import isnan

def compute_insulation(matrix, w=5, triple=False, cooler=False):
    """
    Compute insulation score across the diagonal of a matrix.

    Parameters
    ----------
    matrix : 2D np.array
        Hi-C interaction matrix.
    w : int
        Window size for computing insulation score.
    triple : bool
        If True, assumes matrix is 3D (not used here).
    cooler : bool
        If True, uses alternative indexing style.

    Returns
    -------
    scores : list of float
        Insulation scores.
    indexes : list of int
        Corresponding bin indices.
    """
    end = len(matrix) - 1
    scores = []
    for i in range(0, end):
        diag = 0
        counter = 0
        for j in range(i, i + w):
            if j >= end:
                break
            idx = j - 2 * counter
            val = 0 if idx < 0 or idx >= matrix.shape[1] else matrix[j, idx]
            if isnan(val):
                val = 0
            diag += val
            counter += 1
        scores.append(diag)
    return scores, list(range(len(scores)))

def insulation_spearmanr(pred_path, target_path):
    """
    Compute Spearman correlation between predicted and target insulation scores.
    """
    pred_data = np.load(pred_path)
    target_data = np.load(target_path)
    scores = []
    for pred, target in zip(pred_data, target_data):
        corr, _ = spearmanr(pred, target)
        if not isnan(corr):
            scores.append(corr)
    return scores

def insulation_pearson(pred_path, target_path):
    """
    Compute Pearson correlation between predicted and target insulation scores.
    """
    pred_data = np.load(pred_path)
    target_data = np.load(target_path)
    scores = []
    for pred, target in zip(pred_data, target_data):
        corr, _ = pearsonr(pred, target)
        if not isnan(corr):
            scores.append(corr)
    return scores

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate insulation score correlations')
    parser.add_argument('--data-root', default='./', help='Root directory for data and results')
    parser.add_argument('--save-dir', default='insulation_score_result', help='Directory to save evaluation results')
    args = parser.parse_args()
    
    folder_path = args.data_root
    
    # Configuration
    cell_types = ["IMR90"]
    model_names = ["ChromNet"]
    chr_data = pd.read_csv(os.path.join(folder_path, "hg38.chrom.sizes.txt"), sep='\t', header=None)
    chr_data.columns = ["chrom_name", "chrom_length"]
    chr_names = chr_data["chrom_name"].values[19:22]
    chr_lens = chr_data["chrom_length"].values[19:22]
    save_dir = os.path.join(folder_path, args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    for cell_type in cell_types:
        pearson_dict = {"methods": []}
        spearman_dict = {"methods": []}
        for chr_name in chr_names:
            pearson_dict[chr_name] = []
            spearman_dict[chr_name] = []

        print(f"=== Evaluating Insulation Score Correlations for Cell Type: {cell_type} ===")
        for model_name in model_names:
            pearson_dict["methods"].append(model_name)
            spearman_dict["methods"].append(model_name)
            for chr_name in chr_names:
                data_dir = os.path.join(folder_path, f"C.Origami-main/analysis_result/hg38_KR/cell_new_insulation_hicplotter_new_dataset_OK/{cell_type}/")
                target_file = os.path.join(data_dir, f"orign_all_insulation_score_{chr_name}.npy")
                pred_file = os.path.join(data_dir, f"{model_name}_all_insulation_score_{chr_name}.npy")

                # Verify files exist
                if not os.path.exists(target_file) or not os.path.exists(pred_file):
                    print(f"WARNING: Files not found for {model_name} - {chr_name}")
                    pearson_dict[chr_name].append(np.nan)
                    spearman_dict[chr_name].append(np.nan)
                    continue

                p_corr = insulation_pearson(pred_file, target_file)
                s_corr = insulation_spearmanr(pred_file, target_file)

                pearson_dict[chr_name].append(np.mean(p_corr))
                spearman_dict[chr_name].append(np.mean(s_corr))

        pearson_df = pd.DataFrame(pearson_dict)
        pearson_df["avg"] = pearson_df[chr_names].mean(axis=1)
        print("Insulation Score Pearson Correlation:")
        print(pearson_df)
        pearson_df.to_csv(os.path.join(save_dir, f"insulation_pearson_{cell_type}.csv"), index=False)

        spearman_df = pd.DataFrame(spearman_dict)
        spearman_df["avg"] = spearman_df[chr_names].mean(axis=1)
        print("Insulation Score Spearman Correlation:")
        print(spearman_df)
        spearman_df.to_csv(os.path.join(save_dir, f"insulation_spearman_{cell_type}.csv"), index=False)

if __name__ == "__main__":
    main()
