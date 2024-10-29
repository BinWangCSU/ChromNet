import torch
import argparse
import sys
import numpy as np
import os
from data.data_feature import SequenceFeature, GenomicFeature
import ChromNet_model
from utils import plot_utils

# --------------------------------------
# Model utility functions
# --------------------------------------

def load_default(model_path):
    """
    Load the default ChromNet model with a pre-trained checkpoint.
    """
    model = get_model('ChromNet')
    load_checkpoint(model, model_path)
    return model

def get_model(model_name, num_genomic_features=7):
    """
    Create a model instance from the given model name.
    """
    num_cell_types = 2  # Set according to the problem (adjust as needed)
    ModelClass = getattr(ChromNet_model, model_name)
    model = ModelClass(num_genomic_features, num_cell_types)
    return model

def load_checkpoint(model, model_path):
    """
    Load the model weights from the checkpoint.
    """
    print(f'Loading model weights from: {model_path}')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model_weights = checkpoint['state_dict']

    # Clean model weight keys (remove 'model.' prefix)
    for key in list(model_weights):
        model_weights[key.replace('model.', '')] = model_weights.pop(key)
    
    model.load_state_dict(model_weights)
    model.eval()  # Set to evaluation mode
    return model

# --------------------------------------
# Data processing and inference functions
# --------------------------------------

def preprocess_default(seq, ctcf, atac):
    """
    Preprocess sequence, CTCF, and ATAC data before feeding into the model.
    """
    seq = torch.tensor(seq).unsqueeze(0)  # Add batch dimension
    ctcf = torch.tensor(np.nan_to_num(ctcf, 0))  # Replace NaN values with 0
    atac_log = torch.tensor(atac)  # ATAC without log normalization (adjust if needed)

    # Combine features into a single input tensor
    features = [ctcf, atac_log]
    features = torch.cat([feat.unsqueeze(0).unsqueeze(2) for feat in features], dim=2)
    inputs = torch.cat([seq, features], dim=2)

    # Move data to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = inputs.to(device)
    return inputs

def load_region(chr_name, start, seq_path, ctcf_path, atac_path, window=2097152):
    """
    Load genomic data for a specific region of the chromosome.
    """
    end = start + window
    seq, ctcf, atac = load_data_default(chr_name, seq_path, ctcf_path, atac_path)
    seq_region, ctcf_region, atac_region = get_data_at_interval(chr_name, start, end, seq, ctcf, atac)
    return seq_region, ctcf_region, atac_region

def load_data_default(chr_name, seq_path, ctcf_path, atac_path):
    """
    Load sequence, CTCF, and ATAC data for a specific chromosome.
    """
    seq_chr_path = os.path.join(seq_path, f'{chr_name}.fa.gz')
    seq = SequenceFeature(path=seq_chr_path)
    ctcf = GenomicFeature(path=ctcf_path, norm='log')
    atac = GenomicFeature(path=atac_path, norm='log')
    return seq, ctcf, atac

def get_data_at_interval(chr_name, start, end, seq, ctcf, atac):
    """
    Slice genomic data for a specified interval within the chromosome.
    """
    seq_region = seq.get(start, end)
    ctcf_region = ctcf.get(chr_name, start, end)
    atac_region = atac.get(chr_name, start, end)
    return seq_region, ctcf_region, atac_region

def prediction_new(model, seq_region, ctcf_region, atac_region):
    """
    Make a prediction using the pre-loaded model and the given genomic data.
    """
    inputs = preprocess_default(seq_region, ctcf_region, atac_region)
    pred = model(inputs)[0].detach().cpu().numpy()
    return pred

# --------------------------------------
# Main prediction logic
# --------------------------------------

def main():
    parser = argparse.ArgumentParser(description='ChromNet Prediction Module.')
    
    # Arguments for input data and model
    parser.add_argument('--out', dest='output_path', default='outputs', help='Output path for storing results (default: %(default)s)')
    parser.add_argument('--celltype', dest='celltype', help='Sample cell type for prediction')
    parser.add_argument('--chr', dest='chr_name', help='Chromosome for prediction', required=True)
    parser.add_argument('--max_len', dest='max_len', type=int, help='max length for prediction (bp)', required=True)
    parser.add_argument('--model', dest='model_path', help='Path to the model checkpoint', required=True)
    parser.add_argument('--seq', dest='seq_path', help='Path to the folder where the sequence .fa.gz files are stored', required=True)
    parser.add_argument('--ctcf', dest='ctcf_path', help='Path to the folder where the CTCF ChIP-seq .bw files are stored', required=True)
    parser.add_argument('--atac', dest='atac_path', help='Path to the folder where the ATAC-seq .bw files are stored', required=True)
    
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    
    # Run the single prediction process
    single_prediction(args.output_path, args.celltype, args.chr_name, args.max_len, args.model_path, args.seq_path, args.ctcf_path, args.atac_path)

def single_prediction(output_path, celltype, chr_name, max_len, model_path, seq_path, ctcf_path, atac_path):
    """
    Perform predictions over chromosome regions using a sliding window approach.
    """
    model = load_default(model_path)
    increment = 262144  # Window sliding step (256 kb)
    data_len = 2097152  # Input window size (2 Mb)
    max_start = max_len - data_len  # Maximum starting point for sliding window
    
    # Iterate through chromosome in increments
    for start in range(0, max_start, increment):
        # Load the genomic data for the current region
        seq_region, ctcf_region, atac_region = load_region(chr_name, start, seq_path, ctcf_path, atac_path)
        # Make predictions
        pred = prediction_new(model, seq_region, ctcf_region, atac_region)
        pred = np.squeeze(pred)  # Remove unnecessary dimensions

        plot = plot_utils.MatrixPlot(output_path, pred, 'prediction', celltype, 
                                     chr_name, start)
        plot.plot()



    # Last window to ensure full coverage
    start = max_len - data_len
    seq_region, ctcf_region, atac_region = load_region(chr_name, start, seq_path, ctcf_path, atac_path)
    pred = prediction_new(model, seq_region, ctcf_region, atac_region)
    pred = np.squeeze(pred)

    plot = plot_utils.MatrixPlot(output_path, pred, 'prediction', celltype, 
                                 chr_name, start)
    plot.plot()


if __name__ == '__main__':
    main()
