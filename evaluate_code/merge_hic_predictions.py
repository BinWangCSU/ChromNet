import numpy as np
import os
import pandas as pd
import argparse
import sys
from scipy.sparse import save_npz, coo_matrix
from tqdm import tqdm

def merge_hic_predictions(predictions, hic_size, submatrix_size=256, stride=32):
    """
    Merge multiple predicted Hi-C submatrices into a full chromosome-scale matrix with overlap correction.

    Args:
        predictions (list of np.ndarray): List of predicted submatrices (shape: submatrix_size x submatrix_size).
        hic_size (int): Target size of the final Hi-C contact matrix.
        submatrix_size (int): Size of each predicted submatrix (default: 256).
        stride (int): Sliding window step (default: 32).

    Returns:
        np.ndarray: Merged Hi-C matrix (shape: hic_size x hic_size).
    """
    merged_hic = np.zeros((hic_size, hic_size))
    overlap_counts = np.zeros((hic_size, hic_size))

    for i, submatrix in enumerate(tqdm(predictions, desc="Merging submatrices")):
        start = i * stride
        end = start + submatrix_size
        if i < len(predictions) - 1:
            merged_hic[start:end, start:end] += submatrix
            overlap_counts[start:end, start:end] += 1
        else:
            merged_hic[-submatrix_size:, -submatrix_size:] += submatrix
            overlap_counts[-submatrix_size:, -submatrix_size:] += 1

    overlap_counts[overlap_counts == 0] = 1  # Avoid division by zero
    merged_hic /= overlap_counts
    return merged_hic


def load_predictions(model_name, model_path, cell_type, chr_name, num_windows, increment, max_start, data_len):
    """
    Load predicted submatrices from disk for a given model and chromosome.

    Args:
        model_name (str): Name of the model.
        model_path (str): Base path to model outputs.
        cell_type (str): Cell type name.
        chr_name (str): Chromosome name.
        num_windows (int): Number of sliding windows.
        increment (int): Step size for sliding window.
        max_start (int): Length of the chromosome.
        data_len (int): Length of submatrix region in bp.

    Returns:
        list or np.ndarray: List of submatrices or array (for 'orign' model).
    """
    predictions = []
    try:
        if model_name == "orign":
            pred_path = os.path.join(model_path, cell_type, f"{chr_name}_submatrix.npy")
            if not os.path.exists(pred_path):
                print(f"ERROR: Original data file not found: {pred_path}")
                return None
            predictions = np.load(pred_path, allow_pickle=True)
        else:
            for idx in tqdm(range(num_windows + 1), desc=f"Loading {model_name} predictions"):
                if idx < num_windows:
                    offset = increment * idx
                else:
                    offset = max_start - data_len
                pred_file = f"{chr_name}_{offset}.npy"
                pred_path = os.path.join(model_path, cell_type, "prediction", "npy", pred_file)
                
                if not os.path.exists(pred_path):
                    print(f"WARNING: Prediction file not found: {pred_path}")
                    continue
                    
                try:
                    pred_matrix = np.load(pred_path)
                    predictions.append(pred_matrix)
                except Exception as e:
                    print(f"ERROR: Failed to load prediction {pred_path}: {str(e)}")
    except Exception as e:
        print(f"[ERROR] Failed to load predictions for {model_name}, {cell_type}, {chr_name}: {e}")
    
    if len(predictions) == 0:
        print(f"WARNING: No predictions loaded for {model_name}, {cell_type}, {chr_name}")
    else:
        print(f"Successfully loaded {len(predictions)} predictions for {model_name}, {cell_type}, {chr_name}")
        
    return predictions


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Merge Hi-C predictions into chromosome-scale matrices")
    parser.add_argument('--folder_path', default='./', help='Root folder path')
    parser.add_argument('--chrom_info_path', help='Path to chromosome size file (default: <folder_path>/hg38.chrom.sizes.txt)')
    parser.add_argument('--output_dir', default='all_chrom_matrix', help='Output directory for merged matrices')
    parser.add_argument('--start_chr', type=int, default=16, help='Start chromosome index (0-based, default: 16 for chr17)')
    parser.add_argument('--end_chr', type=int, default=22, help='End chromosome index (0-based, default: 22 for chr22)')
    args = parser.parse_args()

    folder_path = args.folder_path
    chrom_info_path = args.chrom_info_path or os.path.join(folder_path, "hg38.chrom.sizes.txt")
    
    # Check if chromosome size file exists
    if not os.path.exists(chrom_info_path):
        print(f"ERROR: Chromosome size file not found: {chrom_info_path}")
        sys.exit(1)
        
    chr_data = pd.read_csv(chrom_info_path, sep="\t", header=None, names=["chrom_name", "chrom_length"])

    # Select chromosomes
    chr_names = chr_data["chrom_name"].values[args.start_chr:args.end_chr]
    chr_lens = chr_data["chrom_length"].values[args.start_chr:args.end_chr]

    model_configs = {
        "orign": os.path.join(folder_path, "orign_submatrix"),
        "ChromNet": os.path.join(folder_path, "outputs_ChromNet")
    }

    increment = 262144
    bin_size = 8192
    data_len = 2097152
    submatrix_size = 256
    stride = 32
    cell_types = ["IMR90"]

    # Create output directory
    output_base_dir = os.path.join(folder_path, args.output_dir)
    os.makedirs(output_base_dir, exist_ok=True)

    for cell_type in cell_types:
        print(f"Processing cell type: {cell_type}")
        for chr_name, chr_len in zip(chr_names, chr_lens):
            print(f"Processing chromosome: {chr_name}")
            hic_matrix_size = int(np.ceil(chr_len / bin_size))
            num_windows = (chr_len - data_len) // increment

            for model_name, model_path in model_configs.items():
                print(f"[INFO] Processing {cell_type} - {chr_name} - {model_name}")
                
                # Ensure output directories exist
                save_dir = os.path.join(output_base_dir, cell_type, model_name)
                os.makedirs(save_dir, exist_ok=True)
                
                predictions = load_predictions(model_name, model_path, cell_type, chr_name, num_windows, increment, chr_len, data_len)
                if predictions is None or len(predictions) == 0:
                    print(f"[SKIP] No predictions available for {cell_type} - {chr_name} - {model_name}")
                    continue

                merged_hic = merge_hic_predictions(predictions, hic_matrix_size, submatrix_size, stride)

                save_path = os.path.join(save_dir, f"{model_name}_matrix_pred_{chr_name}.npz")
                save_npz(save_path, coo_matrix(merged_hic))
                print(f"[SAVED] {save_path} | shape: {merged_hic.shape}")
                
    print("Merge process completed successfully!")

if __name__ == "__main__":
    main()
