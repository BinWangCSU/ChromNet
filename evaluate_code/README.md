# ChromNet: Multi-Task Learning Framework for 3D Chromatin Structure Prediction

## Evaluate ChromNet

4. **Prediction for Multiple Chromosomes**:
   
   ```bash
   python run_test_all_chrom.py
   ```
   
5. **Merge Predictions**:
   ```bash
   python merge_hic_predictions.py --folder_path ./ --output_dir all_chrom_matrix
   ```

6. **Calculate Insulation Scores**:
   ```bash
   python insulation_score_cal.py
   ```

7. **Evaluate Results**:
   ```bash
   python evaluate_insulation_score_correlations.py --data-root ./ --save-dir insulation_score_result
   python distance_stratified_correlation.py --data_root ./ --save_path ./correlation_results
   ```

8. **Visualize Results**:
   ```bash
   python plot_sub_matrix.py --data_root ./ --output_dir plot_submatrix
   ```

## Visualization

### Plotting Submatrices

To visualize the prediction results together with epigenetic signals and insulation scores:

```bash
python plot_sub_matrix.py --data_root ./ --output_dir plot_submatrix
```

This script generates comprehensive visualizations showing:
- Original experimental Hi-C matrices
- Predicted Hi-C matrices
- CTCF binding signals
- ATAC-seq signals
- Insulation score comparisons

#### Visualization Options

- `--data_root`: Root directory containing data files (default: './')
- `--output_dir`: Directory to save plots (default: 'plot_submatrix')
- `--chr_info`: Chromosome size information file (default: 'hg38.chrom.sizes.txt')
- `--cell_types`: Cell types to process (default: ['IMR90'])
- `--model_names`: Model names to process (default: ['ChromNet'])
- `--chr_indices`: Indices of chromosomes to process (default: [1, 9, 14])

For more details, run:
```bash
python plot_sub_matrix.py --help
```