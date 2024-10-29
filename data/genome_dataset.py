import pandas as pd
from torch.utils.data import Dataset
from data.chromosome_dataset import ChromosomeDataset
import data.data_feature as data_feature

class GenomeDataset(Dataset):
    '''
    Load all chromosomes
    '''
    def __init__(self, cell_types,
                       celltype_root, 
                       genome_assembly,
                       feat_dicts, 
                       mode = 'train', 
                       include_sequence = True,
                       include_genomic_features = True,
                       use_aug = False):
        self.cell_types = cell_types
        self.data_root = celltype_root
        self.include_sequence = include_sequence
        self.include_genomic_features = include_genomic_features
        self.feat_dicts=feat_dicts

        # Create a label mapping for cell types
        self.label_mapping = {cell_type: idx for idx, cell_type in enumerate(self.cell_types)}

        if not self.include_sequence: print('Not using sequence!')
        if not self.include_genomic_features: print('Not using genomic features!')
        self.use_aug = use_aug

        if mode != 'train': self.use_aug = False # Set augmentation

        self.chr_names = self.get_chr_names(genome_assembly)

        if genome_assembly=="mm10":
            if mode == 'train':
                self.chr_names = ['chr%s'%(x) for x in range(1,16)]
            elif mode == 'val':
                self.chr_names = ['chr%s'%(x) for x in range(16,18)]
            elif mode == 'test':
                self.chr_names = ['chr%s'%(x) for x in range(18,20)]
            else:
                raise Exception(f'Unknown mode {mode}')

        else:

            if mode == 'train':
                self.chr_names = ['chr%s'%(x) for x in range(1,17)]
            elif mode == 'val':
                self.chr_names = ['chr%s'%(x) for x in range(17,20)]
            elif mode == 'test':
                self.chr_names = ['chr%s'%(x) for x in range(20,23)]
            else:
                raise Exception(f'Unknown mode {mode}')
            
        print(self.chr_names)

        self.chr_data, self.lengths = self.load_chrs(self.chr_names, self.data_root, self.feat_dicts, self.cell_types)

        self.ranges = self.get_ranges(self.lengths)

    def __getitem__(self, idx):
        cell_type, chr_name, chr_idx = self.get_chr_idx(idx)
        seq, features, mat, start, end = self.chr_data[cell_type][chr_name][chr_idx]

        label = self.label_mapping[cell_type]  # Convert cell_type to label

        if self.include_sequence:
            if self.include_genomic_features: # Both
                outputs = seq, features, mat, start, end, label, chr_name, chr_idx
            else: # sequence only  
                outputs = seq, mat, start, end, label, chr_name, chr_idx
        else: 
            if self.include_genomic_features: # features only
                outputs = features, mat, start, end, label, chr_name, chr_idx
            else: raise Exception('Must have at least one of the sequence or features')
        return outputs

    def __len__(self):
        total_length = 0
        for cell_type_lengths in self.lengths.values():
            total_length += sum(cell_type_lengths)
        return total_length
    
    def load_chrs(self, chr_names, data_root, feat_dicts, cell_types):
        print('Loading chromosome datasets...')
        chr_data_dict = {}
        lengths = {}
        for idx in range(len(data_root)):
            celltype_roots=data_root[idx]
            feat_dict=feat_dicts[idx]
            cell_type=cell_types[idx]
            if cell_type=="IMR90_noise":
                self.use_aug=True
                celltype_roots = celltype_roots.replace("_noise", "")
            genomic_features = self.load_features(f'{celltype_roots}/genomic_features', feat_dict)
            chr_data_dict[cell_type] = {}
            lengths[cell_type] = []
            print(chr_names)
            for chr_name in chr_names:
                chr_data_dict[cell_type][chr_name] = ChromosomeDataset(celltype_roots, chr_name, genomic_features, self.use_aug)
                lengths[cell_type].append(len(chr_data_dict[cell_type][chr_name]))
        print('Chromosome datasets loaded')
        return chr_data_dict, lengths

    def load_features(self, root_dir, feat_dicts):
        feat_list = []
        for feat_item in list(feat_dicts.values()):
            file_name = feat_item['file_name']
            file_path = f'{root_dir}/{file_name}'
            norm = feat_item['norm']
            feat_list.append(data_feature.GenomicFeature(file_path, norm))
        return feat_list
        
    def get_chr_names(self, assembly):
        print(f'Using Assembly: {assembly}')
        if assembly in ['hg38', 'hg19']:
            chrs = list(range(1, 23))
        elif assembly in ['mm10', 'mm9']:
            chrs = list(range(1, 20))
        else: raise Exception(f'Assembly {assembly} unknown')
        chrs.append('X')
        #chrs.append('Y')
        chr_names = []
        for chr_num in chrs:
            chr_names.append(f'chr{chr_num}')
        return chr_names

    def get_ranges(self, lengths):
        current_start = {}
        ranges = {}
        for cell_type, cell_lengths in lengths.items():
            current_start[cell_type] = 0
            ranges[cell_type] = []
            for length in cell_lengths:
                ranges[cell_type].append([current_start[cell_type], current_start[cell_type] + length - 1])
                current_start[cell_type] += length
        return ranges

    def get_chr_idx(self, idx):
        current_start = 0
        for cell_type, cell_type_lengths in self.lengths.items():
            for i, length in enumerate(cell_type_lengths):
                start = current_start
                end = start + length - 1
                if start <= idx <= end:
                    return cell_type, self.chr_names[i], idx - start
                current_start += length


    def proc_centrotelo(self, bed_dir):
        df = pd.read_csv(bed_dir , sep = '\t', names = ['chr', 'start', 'end'])
        chrs = df['chr'].unique()
        centrotelo_dict = {}
        for chr_name in chrs:
            sub_df = df[df['chr'] == chr_name]
            regions = sub_df.drop('chr', axis = 1).to_numpy()
            centrotelo_dict[chr_name] = regions
        return centrotelo_dict

if __name__ == '__main__':
    main()
