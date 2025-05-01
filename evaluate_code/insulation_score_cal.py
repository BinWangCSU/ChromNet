import numpy as np
import os
import pandas as pd
from math import isnan

# -------------------------------
# Configuration
# -------------------------------
folder_path = "./"
insulation_window = 5
relative_window = 8
bin_size = 8192
data_len = 2097152
increment = 262144

cell_types = ["IMR90"]
model_names = ["orign", "ChromNet"]

# Load chromosome information
chr_data = pd.read_csv(os.path.join(folder_path, "hg38.chrom.sizes.txt"), sep='\t', header=None)
chr_data.columns = ["chrom_name", "chrom_length"]
chr_names = chr_data["chrom_name"].values[19:22]
chr_lens = chr_data["chrom_length"].values[19:22]

# -------------------------------
# Insulation score calculation
# -------------------------------
def compute_insulation(matrix, window=5, triple=False, cooler=False):
    """
    Calculate insulation scores along the diagonal of a matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Hi-C contact matrix.
    window : int
        Window size along the diagonal.

    Returns
    -------
    scores : list of float
        Insulation scores.
    indexes : list of int
        Corresponding bin positions.
    """
    scores = []
    indexes = []
    end = len(matrix) - 1
    for i in range(0, end):
        diag_sum = 0
        counter = 0
        for j in range(i, i + window):
            if j >= end:
                break
            idx = j - 2 * counter
            value = matrix[j, idx] if 0 <= idx < matrix.shape[1] else 0
            if isnan(value):
                value = 0
            diag_sum += value
            counter += 1
        scores.append(diag_sum)
        indexes.append(i)
    return scores, indexes

# -------------------------------
# Main loop
# -------------------------------
for cell_type in cell_types:
    save_dir = os.path.join(folder_path, f"insulation_hicplotter_cal_result/{cell_type}/")
    os.makedirs(save_dir, exist_ok=True)

    for chr_idx, chr_name in enumerate(chr_names):
        chr_len = chr_lens[chr_idx]
        num_windows = (chr_len - data_len) // increment

        for model_name in model_names:
            insulation_scores = []
            insulator_positions = []

            if model_name == "orign":
                matrix_path = os.path.join(folder_path, f"orign_submatrix/{cell_type}/{chr_name}_submatrix.npy")
                all_pred_matrix = np.load(matrix_path)

                for idx in range(num_windows + 2):
                    pred_matrix = all_pred_matrix[idx]
                    scores, tricks = compute_insulation(pred_matrix, insulation_window)
                    insulation_scores.append(scores)
                    insulator_positions.append(tricks)

                np.save(os.path.join(save_dir, f"orign_all_insulation_score_{chr_name}.npy"), insulation_scores)
                np.save(os.path.join(save_dir, f"orign_all_tricks_{chr_name}.npy"), insulator_positions)

            else:
                model_path = os.path.join(folder_path, f"outputs_{model_name}/{cell_type}/prediction/npy/")
                for idx in range(num_windows + 1):
                    file_path = os.path.join(model_path, f"{chr_name}_{increment * idx}.npy")
                    pred_matrix = np.load(file_path)
                    scores, tricks = compute_insulation(pred_matrix, insulation_window)
                    insulation_scores.append(scores)
                    insulator_positions.append(tricks)

                # Final patch (last segment)
                last_path = os.path.join(model_path, f"{chr_name}_{chr_len - data_len}.npy")
                pred_matrix = np.load(last_path)
                scores, tricks = compute_insulation(pred_matrix, insulation_window)
                insulation_scores.append(scores)
                insulator_positions.append(tricks)

                np.save(os.path.join(save_dir, f"{model_name}_all_insulation_score_{chr_name}.npy"), insulation_scores)
                np.save(os.path.join(save_dir, f"{model_name}_all_tricks_{chr_name}.npy"), insulator_positions)
