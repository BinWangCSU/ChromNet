import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import sys
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import pyBigWig
import matplotlib.gridspec as gridspec

from tqdm import tqdm

# Configure color map
color_map = LinearSegmentedColormap.from_list("bright_red", [(1, 1, 1), (1, 0, 0)])

def load_insulation_score(file_path):
    """
    Load insulation score from npy file
    
    Args:
        file_path (str): Path to the insulation score file
        
    Returns:
        numpy.ndarray: Loaded insulation score data
    """
    try:
        if not os.path.exists(file_path):
            print(f"ERROR: Insulation score file {file_path} not found")
            return None
        return np.load(file_path)
    except Exception as e:
        print(f"ERROR: Failed to load insulation score from {file_path}: {e}")
        return None

def load_bigwig_data(bw_file, chr_name, start_pos, end_pos, bin_size):
    """
    Load and process CTCF or ATAC-seq data from bigwig file
    
    Args:
        bw_file (str): Path to the bigwig file
        chr_name (str): Chromosome name
        start_pos (int): Start position on chromosome
        end_pos (int): End position on chromosome
        bin_size (int): Size of each bin
        
    Returns:
        numpy.ndarray: Binned signal values
    """
    try:
        if not os.path.exists(bw_file):
            print(f"ERROR: Bigwig file {bw_file} not found")
            return np.zeros(len(np.arange(start_pos, end_pos, bin_size))-1)
            
        bw = pyBigWig.open(bw_file)
        # Ensure chromosome name format is consistent
        if not chr_name.startswith('chr') and 'chr' + chr_name in bw.chroms():
            chr_name = 'chr' + chr_name
        
        # Prepare needed bins
        bins = np.arange(start_pos, end_pos, bin_size)
        values = []
        
        # Get average signal value for each bin
        for i in range(len(bins)-1):
            try:
                val = bw.stats(chr_name, int(bins[i]), int(bins[i+1]), type="mean")[0]
                values.append(0 if val is None else val)
            except Exception as e:
                print(f"Warning: Error getting stats for {chr_name}:{bins[i]}-{bins[i+1]}: {e}")
                values.append(0)  # Use 0 value if error occurs
        
        bw.close()
        return np.array(values)
    except Exception as e:
        print(f"Error loading bigwig file {bw_file}: {e}")
        # Return zero array as substitute in case of error
        return np.zeros(len(np.arange(start_pos, end_pos, bin_size))-1)


def plot_and_save_matrices(predicted_matrix, true_matrix, ctcf_data, atac_data, 
                           ins_score_pred_sub, ins_score_exp_sub, save_dir, idx, 
                           comp_cell_type, chr_name, increment, bin_size, model_name):
    """
    Plot and save comparison figures for HiC matrices and associated data
    
    Args:
        predicted_matrix (numpy.ndarray): Predicted HiC matrix
        true_matrix (numpy.ndarray): True HiC matrix
        ctcf_data (numpy.ndarray): CTCF signal data
        atac_data (numpy.ndarray): ATAC-seq signal data
        ins_score_pred_sub (numpy.ndarray): Predicted insulation score
        ins_score_exp_sub (numpy.ndarray): Experimental insulation score
        save_dir (str): Directory to save output figures
        idx (int): Index/Position number
        comp_cell_type (str): Cell type name
        chr_name (str): Chromosome name
        increment (int): Window increment size
        bin_size (int): Bin size for the matrix
        model_name (str): Name of the model used for prediction
    """
    try:
        # Calculate positions (Mb)
        start_pos = idx * increment
        end_pos = start_pos + predicted_matrix.shape[0] * bin_size
        start_pos_mb = start_pos / 1e6
        end_pos_mb = end_pos / 1e6

        fig = plt.figure(figsize=(5, 14))  # Height can be adjusted
        gs = gridspec.GridSpec(5, 1, height_ratios=[5, 5, 1.2, 1.2, 1.5])

        # True Hi-C
        ax1 = plt.subplot(gs[0])
        im1 = ax1.imshow(true_matrix, cmap=color_map)
        ax1.set_title(f'Experiment Hi-C ({comp_cell_type})')
        ax1.set_xticks([0, predicted_matrix.shape[1] // 2, predicted_matrix.shape[1] - 1])
        ax1.set_xticklabels([f'{start_pos_mb:.1f} Mb', f'{(start_pos_mb + end_pos_mb) / 2:.1f} Mb', f'{end_pos_mb:.1f} Mb'])
        ax1.set_yticks([0, predicted_matrix.shape[0] // 2, predicted_matrix.shape[0] - 1])
        ax1.set_yticklabels([f'{start_pos_mb:.1f} Mb', f'{(start_pos_mb + end_pos_mb) / 2:.1f} Mb', f'{end_pos_mb:.1f} Mb'])
        plt.colorbar(im1, ax=ax1, shrink=0.6)

        # Predicted Hi-C
        ax2 = plt.subplot(gs[1])
        im2 = ax2.imshow(predicted_matrix, cmap=color_map)
        ax2.set_title(f'{model_name} prediction Hi-C ({comp_cell_type})')
        ax2.set_xticks([0, true_matrix.shape[1] // 2, true_matrix.shape[1] - 1])
        ax2.set_xticklabels([f'{start_pos_mb:.1f} Mb', f'{(start_pos_mb + end_pos_mb) / 2:.1f} Mb', f'{end_pos_mb:.1f} Mb'])
        ax2.set_yticks([0, true_matrix.shape[0] // 2, true_matrix.shape[0] - 1])
        ax2.set_yticklabels([f'{start_pos_mb:.1f} Mb', f'{(start_pos_mb + end_pos_mb) / 2:.1f} Mb', f'{end_pos_mb:.1f} Mb'])
        plt.colorbar(im2, ax=ax2, shrink=0.6)

        # CTCF signal
        ax3 = plt.subplot(gs[2])
        x = np.arange(len(ctcf_data))
        ax3.fill_between(x, ctcf_data, color='steelblue', alpha=0.8, linewidth=0)
        ax3.set_title('CTCF', fontsize=9, loc='left')
        ax3.set_xlim(0, len(ctcf_data) - 1)
        ax3.set_xticks([0, predicted_matrix.shape[1] // 2, predicted_matrix.shape[1] - 1])
        ax3.set_xticklabels([f'{start_pos_mb:.1f} Mb', f'{(start_pos_mb + end_pos_mb) / 2:.1f} Mb', f'{end_pos_mb:.1f} Mb'], fontsize=7)
        ax3.set_yticks([])
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.spines['left'].set_visible(False)

        # ATAC signal
        ax4 = plt.subplot(gs[3])
        x = np.arange(len(atac_data))
        ax4.fill_between(x, atac_data, color='red', alpha=0.8, linewidth=0)
        ax4.set_title('ATAC', fontsize=9, loc='left')
        ax4.set_xlim(0, len(atac_data) - 1)
        ax4.set_xticks([0, predicted_matrix.shape[1] // 2, predicted_matrix.shape[1] - 1])
        ax4.set_xticklabels([f'{start_pos_mb:.1f} Mb', f'{(start_pos_mb + end_pos_mb) / 2:.1f} Mb', f'{end_pos_mb:.1f} Mb'], fontsize=7)
        ax4.set_yticks([])
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        ax4.spines['left'].set_visible(False)

        # Insulation score
        ax5 = plt.subplot(gs[4])
        valid_indices = ~np.isnan(ins_score_exp_sub) & ~np.isnan(ins_score_pred_sub)
        if np.sum(valid_indices) > 1:
            corr = np.corrcoef(ins_score_exp_sub[valid_indices], ins_score_pred_sub[valid_indices])[0, 1]
            ax5.text(0.7, 0.8, f"r = {corr:.2f}", transform=ax5.transAxes, fontsize=7)

        ins_exp_norm = (ins_score_exp_sub - np.nanmean(ins_score_exp_sub)) / np.nanstd(ins_score_exp_sub)
        ins_pred_norm = (ins_score_pred_sub - np.nanmean(ins_score_pred_sub)) / np.nanstd(ins_score_pred_sub)

        ax5.plot(range(len(ins_exp_norm)), ins_exp_norm, 'k-', linewidth=0.8, label='Target')
        ax5.plot(range(len(ins_pred_norm)), ins_pred_norm, 'k--', linewidth=0.8, label='Prediction')
        ax5.set_ylabel("Insulation\ncomparison", fontsize=8)
        ax5.legend(loc='upper left', fontsize=6, frameon=False)
        ax5.set_ylim(-4, 4)
        ax5.set_xticks([0, predicted_matrix.shape[1] // 2, predicted_matrix.shape[1] - 1])
        ax5.set_xticklabels([f'{start_pos_mb:.1f} Mb', f'{(start_pos_mb + end_pos_mb) / 2:.1f} Mb', f'{end_pos_mb:.1f} Mb'])
        ax5.spines['top'].set_visible(False)
        ax5.spines['right'].set_visible(False)

        # Remove blank margins
        plt.tight_layout(pad=0.2)

        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        # Save the figure
        output_file = os.path.join(save_dir, f"all_sub_matrix_{comp_cell_type}_{chr_name}_{idx}.png")
        plt.savefig(output_file)
        plt.close()
        
        return output_file
    except Exception as e:
        print(f"Error plotting matrices: {e}")
        plt.close()
        return None

def main():
    """
    Main function to handle command line arguments and run the plotting process
    """
    parser = argparse.ArgumentParser(description='Plot submatrices with insulation score and epigenetic data')
    parser.add_argument('--data_root', type=str, default='./', help='Root directory for data')
    parser.add_argument('--output_dir', type=str, default='plot_submatrix', help='Output directory for plots')
    parser.add_argument('--chr_info', type=str, default='hg38.chrom.sizes.txt', help='Chromosome size information file')
    parser.add_argument('--cell_types', nargs='+', default=['IMR90'], help='Cell types to process')
    parser.add_argument('--model_names', nargs='+', default=['ChromNet'], help='Model names to process')
    parser.add_argument('--chr_indices', nargs='+', type=int, default=[19, 20, 21], 
                        help='Indices of chromosomes to process (0-based)')
    args = parser.parse_args()
    
    # Set parameters
    data_root = args.data_root
    output_base_dir = os.path.join(data_root, args.output_dir)
    chr_info_path = os.path.join(data_root, args.chr_info)
    cell_types = args.cell_types
    model_names = args.model_names
    chr_indices = args.chr_indices
    
    # Check if chromosome info file exists
    if not os.path.exists(chr_info_path):
        print(f"ERROR: Chromosome info file {chr_info_path} not found")
        sys.exit(1)
    
    # Load chromosome data
    try:
        chr_data = pd.read_csv(chr_info_path, sep='\t', header=None)
        chr_data.columns = ["chrom_name", "chrom_length"]
        all_chr_names = chr_data["chrom_name"].values
        all_chr_lens = chr_data["chrom_length"].values
        
        chr_names = [all_chr_names[i] for i in chr_indices if i < len(all_chr_names)]
        chr_lens = [all_chr_lens[i] for i in chr_indices if i < len(all_chr_lens)]
        
        if not chr_names:
            print("ERROR: No valid chromosomes selected")
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR: Failed to load chromosome data: {e}")
        sys.exit(1)
    
    # Parameter settings
    increment = 262144  # Window increment size
    bin_size = 8192
    data_len = 2097152
    
    # Process each cell type and model
    for cell_type in cell_types:
        print(f"Processing cell type: {cell_type}")
        for model_name in model_names:
            print(f"Processing model: {model_name}")
            
            save_dir = os.path.join(output_base_dir, cell_type, model_name)
            os.makedirs(save_dir, exist_ok=True)
            
            for chr_idx, (chr_name, chr_len) in enumerate(zip(chr_names, chr_lens)):
                max_start = chr_len
                num_windows = (max_start - data_len) // increment
                print(f"Processing chromosome: {chr_name} (length: {chr_len})")
                
                # Path settings
                model_path_ori = os.path.join(data_root, "orign_submatrix")
                model_path = os.path.join(data_root, f"outputs_{model_name}")
                
                # Check if paths exist
                if not os.path.exists(model_path_ori):
                    print(f"WARNING: Original matrix path {model_path_ori} not found")
                    continue
                    
                if not os.path.exists(model_path):
                    print(f"WARNING: Model output path {model_path} not found")
                    continue
                
                # Load original Hi-C matrix
                true_matrix_path = os.path.join(model_path_ori, f"{cell_type}/{chr_name}_submatrix.npy")
                if not os.path.exists(true_matrix_path):
                    print(f"WARNING: True matrix file {true_matrix_path} not found")
                    continue
                    
                try:
                    true_matrix = np.load(true_matrix_path)
                except Exception as e:
                    print(f"ERROR: Failed to load true matrix: {e}")
                    continue
                
                # Load prediction matrices
                predictions = []
                try:
                    for window_idx in tqdm(range(num_windows + 1), desc=f"Loading prediction windows for {chr_name}"):
                        pred_file = os.path.join(model_path, f"{cell_type}/prediction/npy/{chr_name}_{increment * window_idx}.npy")
                        if not os.path.exists(pred_file):
                            print(f"WARNING: Prediction file {pred_file} not found")
                            continue
                        pred_matrix = np.load(pred_file)
                        predictions.append(pred_matrix)
                    
                    # Load the last window
                    last_pred_file = os.path.join(model_path, f"{cell_type}/prediction/npy/{chr_name}_{max_start - data_len}.npy")
                    if os.path.exists(last_pred_file):
                        pred_matrix = np.load(last_pred_file)
                        predictions.append(pred_matrix)
                    
                    if not predictions:
                        print(f"WARNING: No predictions loaded for {chr_name}")
                        continue
                        
                except Exception as e:
                    print(f"ERROR: Failed to load predictions: {e}")
                    continue
                
                # Load CTCF and ATAC data
                ctcf_data = []
                atac_data = []
                for ca_idx in tqdm(range(num_windows + 1), desc=f"Loading epigenetic data for {chr_name}"):
                    start_pos = increment * ca_idx
                    end_pos = start_pos + data_len
                    
                    ctcf_file = os.path.join(data_root, f"train_data/hg38/{cell_type}/genomic_features/ctcf.bw")
                    atac_file = os.path.join(data_root, f"train_data/hg38/{cell_type}/genomic_features/atac.bw")
                    
                    if not os.path.exists(ctcf_file) or not os.path.exists(atac_file):
                        print(f"WARNING: Epigenetic data files not found")
                        ctcf_data.append(np.zeros(256))
                        atac_data.append(np.zeros(256))
                        continue
                    
                    ctcf_comp = load_bigwig_data(ctcf_file, chr_name, start_pos, end_pos, bin_size)
                    atac_comp = load_bigwig_data(atac_file, chr_name, start_pos, end_pos, bin_size)
                    
                    ctcf_data.append(ctcf_comp)
                    atac_data.append(atac_comp)
                
                # Load last window epigenetic data
                ctcf_file = os.path.join(data_root, f"train_data/hg38/{cell_type}/genomic_features/ctcf.bw")
                atac_file = os.path.join(data_root, f"train_data/hg38/{cell_type}/genomic_features/atac.bw")
                
                if os.path.exists(ctcf_file) and os.path.exists(atac_file):
                    ctcf_comp = load_bigwig_data(ctcf_file, chr_name, max_start - data_len, max_start, bin_size)
                    atac_comp = load_bigwig_data(atac_file, chr_name, max_start - data_len, max_start, bin_size)
                    ctcf_data.append(ctcf_comp)
                    atac_data.append(atac_comp)
                
                # Load insulation score
                ins_score_pred_path = os.path.join(data_root, f"insulation_hicplotter_cal_result/{cell_type}/{model_name}_all_insulation_score_{chr_name}.npy")
                ins_score_exp_path = os.path.join(data_root, f"insulation_hicplotter_cal_result/{cell_type}/orign_all_insulation_score_{chr_name}.npy")
                
                if not os.path.exists(ins_score_pred_path) or not os.path.exists(ins_score_exp_path):
                    print(f"WARNING: Insulation score files not found")
                    continue
                
                ins_score_pred = load_insulation_score(ins_score_pred_path)
                ins_score_exp = load_insulation_score(ins_score_exp_path)
                
                if ins_score_pred is None or ins_score_exp is None:
                    print(f"WARNING: Failed to load insulation scores")
                    continue
                
                # Process each window's insulation score
                ins_score_pred_all = []
                ins_score_pred_exp = []
                
                for ca_idx in tqdm(range(num_windows + 1), desc=f"Processing insulation scores for {chr_name}"):
                    start_pos = increment * ca_idx
                    end_pos = start_pos + data_len
                    start_bin = int(start_pos / bin_size)
                    end_bin = int(end_pos / bin_size)
                    
                    if start_bin >= len(ins_score_pred) or end_bin > len(ins_score_pred):
                        print(f"WARNING: Insulation score index out of range at window {ca_idx}")
                        ins_score_pred_all.append(np.zeros(end_bin - start_bin - 1))
                        ins_score_pred_exp.append(np.zeros(end_bin - start_bin - 1))
                        continue
                        
                    ins_score_pred_sub = ins_score_pred[start_bin:end_bin-1]
                    ins_score_exp_sub = ins_score_exp[start_bin:end_bin-1]
                    
                    ins_score_pred_all.append(ins_score_pred_sub)
                    ins_score_pred_exp.append(ins_score_exp_sub)
                
                # Process last window insulation score
                try:
                    last_start_bin = int((max_start - data_len) / bin_size)
                    last_end_bin = int(max_start / bin_size)
                    
                    if last_end_bin <= len(ins_score_pred):
                        ins_score_pred_sub = ins_score_pred[last_start_bin:last_end_bin-1]
                        ins_score_exp_sub = ins_score_exp[last_start_bin:last_end_bin-1]
                    else:
                        ins_score_pred_sub = ins_score_pred[len(ins_score_pred)-end_bin+1:]
                        ins_score_exp_sub = ins_score_exp[len(ins_score_pred)-end_bin+1:]
                    
                    ins_score_pred_all.append(ins_score_pred_sub)
                    ins_score_pred_exp.append(ins_score_exp_sub)
                except Exception as e:
                    print(f"WARNING: Error processing last window insulation scores: {e}")
                
                # Plot and save each window's chart
                print(f"Plotting {len(predictions)} predicted matrices for {chr_name}")
                for i, pred_matrix in enumerate(tqdm(predictions, desc=f"Plotting matrices for {chr_name}")):
                    if i >= len(ctcf_data) or i >= len(atac_data) or i >= len(ins_score_pred_all) or i >= len(ins_score_pred_exp):
                        print(f"WARNING: Data index mismatch at window {i}")
                        continue
                        
                    if i >= len(true_matrix):
                        print(f"WARNING: True matrix index out of range at window {i}")
                        continue
                        
                    try:
                        true_mat = true_matrix[i]
                        output_file = plot_and_save_matrices(
                            pred_matrix, true_mat, ctcf_data[i], atac_data[i], 
                            ins_score_pred_all[i], ins_score_pred_exp[i], 
                            save_dir, i, cell_type, chr_name, increment, bin_size, model_name
                        )
                        if output_file:
                            print(f"Saved plot to {output_file}")
                    except Exception as e:
                        print(f"ERROR: Failed to plot matrix {i}: {e}")
    
    print("Processing completed successfully!")

if __name__ == "__main__":
    main()
        
