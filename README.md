# ChromNet: Multi-Task Learning Framework for 3D Chromatin Structure Prediction

**ChromNet** is a multi-task learning framework designed for predicting 3D chromatin interactions across different cell types. The model leverages nucleotide-level DNA sequence features and cell-type-specific epigenetic signals, such as CTCF ChIP-seq and ATAC-seq, to predict chromatin interaction matrices. It also incorporates noise perturbation strategies to enhance generalization across cell types.



## Installation

### Prerequisites

Make sure you have the following installed:

- Python 3.9.9
- PyTorch 1.10.2
- PyTorch Lightning 1.9.5
- Other dependencies: Install using the `environment.yml` file.

```bash
conda env create -f environment.yml
```

### Dataset Preparation

You will need to collect and preprocess epigenetic and DNA sequence data for different cell types. Here's an example of how to structure your data:

- **Epigenetic Data**: CTCF ChIP-seq, ATAC-seq (BigWig format)
- **DNA Sequence Data**: Human reference genome (hg38)

Example directory structure:

```text
ChromNet/
    train_data/
      hg38/
		dna_sequence/
			chr1.fa.gz
			chr2.fa.gz
			...
		IMR90/
			hic_matrix/
				chr1.npz
				chr2.npz
				...
			genomic_features/
                  ctcf.bw
                  atac.bw


```

#### Preprocessed Data and Model Weights Download

You can download the preprocessed training dataset for IMR90 and the trained model weights using the following links:

- **Preprocessed Training Dataset for IMR90**: [Download here](https://drive.google.com/file/d/1qderzhKB9yQK2NmQrcVOYTjFLG2ycnkG/view?usp=drive_link)
- **Trained Model Weights**: [Download here](https://drive.google.com/file/d/1OG9VW8G_zjCO_a-IVuCU-juOsLfbch8r/view?usp=drive_link)

## Training

### Command-Line Arguments

Use the following command-line arguments to train the model:

- `--data-root`: Path to the root directory containing the datasets.
- `--celltype`: List of cell types (e.g., IMR90, K562, GM12878).
- `--max-epochs`: Maximum number of training epochs.
- `--batch-size`: Batch size for training.
- `--num-gpu`: Number of GPUs to use.

Example command:

```bash
python train_model.py --save_path checkpoint_ChromNet/ --data-root train_data/ --assembly hg38 --celltype IMR90 K562 IMR90_noise --num-gpu 1 --batch-size 4  --save-top-n 1
```

### Training Script Overview

- The script `train_model.py` initializes the model and data loaders, and trains **ChromNet** using multi-task learning to predict chromatin interactions and classify cell types.
- **Early Stopping** and **Checkpointing**: The training process uses early stopping to prevent overfitting and saves the best-performing model checkpoints.

### Preprocessing CTCF and ATAC-Seq Data

To preprocess CTCF and ATAC-seq data, use the provided bash script in `scripts/`. This script merges, sorts, and scales your BAM files before converting them to BigWig format using `bamCoverage`.

```bash
bash preprocess_epig_data.sh <output_folder> <sample_name>  <bam_folder>
```

## Model Inference

Once trained, ChromNet can be used for predicting chromatin interactions in novel cell types. Use the following command to make predictions:

```bash
python predict.py --out ./outputs --celltype IMR90 --chr chr20 --max_len 64444167 --model ./checkpoints/ChromNet_model_weights.ckpt --seq ./train_data/hg38/dna_sequence --ctcf ./train_data/hg38/IMR90/genomic_features/ctcf.bw --atac ./train_data/hg38/IMR90/genomic_features/atac.bw
```

### Prediction Options

- `--out`: Output directory to store the predictions.
- `--celltype`: Cell type for prediction.
- `--chr`: Chromosome for prediction (e.g., `chr20`).
- `--max_len`: Max length (bp) for prediction.
- `--model`: Path to the pre-trained model checkpoint.
- `--seq`: Path to the DNA sequence `.fa.gz` files.
- `--ctcf`: Path to the CTCF `.bw` files.
- `--atac`: Path to the ATAC-seq `.bw` files.