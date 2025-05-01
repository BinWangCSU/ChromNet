# distance_stratified_correlation

import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy.sparse import load_npz
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm
import sys

def compute_distance_stratified_correlation(preds, targets, method="pearson"):
    distance_list = []
    for dis in tqdm(range(256), desc=f"Computing {method} correlation", leave=False):
        pred_diag = np.diagonal(preds, offset=dis)
        target_diag = np.diagonal(targets, offset=dis)
        mask = ~np.logical_or(np.isnan(pred_diag), np.isnan(target_diag))
        if np.sum(mask) < 2:
            break
        if method == "pearson":
            metric, _ = pearsonr(pred_diag[mask], target_diag[mask])
        elif method == "spearman":
            metric, _ = spearmanr(pred_diag[mask], target_diag[mask])
        else:
            raise ValueError("Unsupported method: choose 'pearson' or 'spearman'")
        distance_list.append(metric)
    return distance_list


def analyze_all_cells(folder_path, output_dir):
    cell_types = ["IMR90"]
    model_names = ["ChromNet"]
    label_names = ["ChromNet"]
    genome_distances = [x * 8192 for x in range(256)]

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Check if chromosome size file exists
    chr_info_path = os.path.join(folder_path, "hg38.chrom.sizes.txt")
    if not os.path.exists(chr_info_path):
        chr_info_path = os.path.join(folder_path, "C.Origami-main/analysis_result/code/hg38.chrom.sizes.txt")
        if not os.path.exists(chr_info_path):
            print(f"ERROR: Could not find chromosome size file at expected locations. Please specify the correct path.")
            sys.exit(1)

    chr_data = pd.read_csv(chr_info_path, sep='\t', header=None, names=["chrom_name", "chrom_length"])
    chr_names = chr_data["chrom_name"].values[19:22]  # chr20–chr22

    pearson_all = {ct: [] for ct in cell_types}
    spearman_all = {ct: [] for ct in cell_types}

    for cell in tqdm(cell_types, desc="Processing Cell Types"):
        pearson_chr_all = []
        spearman_chr_all = []

        for chr_name in tqdm(chr_names, desc=f"Processing {cell} Chromosomes"):
            pearson_tmp = []
            spearman_tmp = []

            for model in model_names:
                pred_path = os.path.join(folder_path, f"all_chrom_matrix/{cell}/{model}/{model}_matrix_pred_{chr_name}.npz")
                true_path = os.path.join(folder_path, f"all_chrom_matrix/{cell}/orign/orign_matrix_pred_{chr_name}.npz")

                # Check if files exist
                if not os.path.exists(pred_path) or not os.path.exists(true_path):
                    print(f"WARNING: Files not found for {cell} - {chr_name} - {model}")
                    # Add NaN values to maintain consistent array lengths
                    pearson_tmp.append([np.nan] * 244)  # Use expected length
                    spearman_tmp.append([np.nan] * 244)
                    continue

                try:
                    pred_matrix = load_npz(pred_path).toarray()
                    true_matrix = load_npz(true_path).toarray()

                    pearson_corr = compute_distance_stratified_correlation(pred_matrix, true_matrix, method="pearson")
                    spearman_corr = compute_distance_stratified_correlation(pred_matrix, true_matrix, method="spearman")

                    pearson_tmp.append(pearson_corr)
                    spearman_tmp.append(spearman_corr)
                except Exception as e:
                    print(f"ERROR processing {cell} - {chr_name} - {model}: {str(e)}")
                    # Add NaN values to maintain consistent array lengths
                    pearson_tmp.append([np.nan] * 244)  # Use expected length
                    spearman_tmp.append([np.nan] * 244)

            pearson_chr_all.append(pearson_tmp)
            spearman_chr_all.append(spearman_tmp)

        # mean over chromosomes
        pearson_all[cell] = np.mean(np.array(pearson_chr_all), axis=0)
        spearman_all[cell] = np.mean(np.array(spearman_chr_all), axis=0)

    plot_results(genome_distances, pearson_all, label_names, model_names, output_dir, title="Pearson", suffix="pearson")
    plot_results(genome_distances, spearman_all, label_names, model_names, output_dir, title="Spearman", suffix="spearman")
    
    print(f"Results saved to {output_dir}")


def plot_results(distances, data_dict, label_names, model_names, save_path, title, suffix):
    colors = ['blue']
    line_styles = ['-']

    plt.figure(figsize=(16, 12))
    for idx, (cell, values) in enumerate(data_dict.items()):
        for m_idx, model_label in enumerate(model_names):
            try:
                y_vals = values[m_idx][2:244]
                plt.plot(distances[2:244], y_vals, linestyle=line_styles[m_idx], color=colors[idx],
                         label=f"{cell}-{label_names[m_idx]}")
            except Exception as e:
                print(f"WARNING: Could not plot {cell}-{model_label}: {str(e)}")

    plt.title(f"Distance-stratified {title} correlation", fontsize=22)
    plt.xlabel('Genome Distance', fontsize=18)
    plt.ylabel(f'{title} r', fontsize=18)
    plt.ylim(0, 1)
    xticks_values = [100000, 500000, 1000000, 1500000, 2000000]
    xticks_labels = ['100kb', '500kb', '1Mb', '1.5Mb', '2Mb']
    plt.xticks(xticks_values, xticks_labels, fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    
    plot_file = os.path.join(save_path, f"{suffix}_genome.png")
    plt.savefig(plot_file)
    print(f"Saved plot to {plot_file}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distance-stratified correlation calculation")
    parser.add_argument('--data_root', type=str, required=True, help="Root folder of prediction and original matrix")
    parser.add_argument('--save_path', type=str, required=True, help="Directory to save plots")
    args = parser.parse_args()

    analyze_all_cells(args.data_root, args.save_path)
