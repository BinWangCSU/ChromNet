import sys 
import os
import random
import pickle
import pandas as pd
import numpy as np
from skimage.transform import resize
from torch.utils.data import Dataset
import data.data_feature as data_feature


class ChromosomeDataset(Dataset):
    def __init__(self, celltype_root, chr_name, feature_list, use_noise = True):
        self.use_noise = use_noise
        self.res = 10000 # 10kb resolution
        self.bins = 209.7152 # 209.7152 bins 2097152 bp
        self.image_scale = 256 # IMPORTANT, scale 210 to 256
        self.sample_bins = 500
        self.stride = 50 # bins
        self.chr_name = chr_name

        print(f'Loading chromosome {chr_name}...')

        self.seq = data_feature.SequenceFeature(path = f'{celltype_root}/../dna_sequence/{chr_name}.fa.gz')
        self.genomic_features = feature_list
        # celltype_root_hic=celltype_root.replace("K562", "IMR90")
        celltype_root_hic = "/".join(celltype_root.split("/")[:-1] + ["IMR90"]) if celltype_root.split("/")[-1] != "IMR90" else celltype_root
        self.mat = data_feature.HiCFeature(path = f'{celltype_root_hic}/hic_matrix/{chr_name}.npz')

        # self.omit_regions = omit_regions
        self.check_length() # Check data length

        self.all_intervals = self.get_active_intervals()
        self.intervals = self.all_intervals

    def __getitem__(self, idx):
        start, end = self.intervals[idx]
        target_size = int(self.bins * self.res)

        # Shift Augmentations
        if self.use_noise: 
            start, end = self.shift_aug(target_size, start, end)
        else:
            start, end = self.shift_fix(target_size, start, end)
        seq, features, mat = self.get_data_at_interval(start, end)

        if self.use_noise:
            seq = seq
            # Genomic features
            noisy_features = [self.gaussian_noise(item, 0.1) for item in features]
            noisy_mats = self.generate_noisy_hic(np.array(mat), np.array(features), noisy_features)
            seq, features, mat = seq, noisy_features, noisy_mats

        return seq, features, mat, start, end

    def __len__(self):
        return len(self.intervals)

    def generate_noisy_hic(self, real_hic, real_features, noisy_features):
        '''
        Generate noisy Hi-C data based on correlation with noisy features
        '''
        real_feat = np.stack(real_features, axis=0).flatten()
        noisy_feat = np.stack(noisy_features, axis=0).flatten()
        correlation = np.corrcoef(real_feat, noisy_feat)[0, 1]
        noisy_hic = real_hic * correlation
        return noisy_hic

    def gaussian_noise(self, inputs, std = 1):
        noise = np.random.randn(*inputs.shape) * std
        outputs = inputs + noise
        return outputs

    def reverse(self, seq, features, mat, chance = 0.5):
        '''
        Reverse sequence and matrix
        '''
        r_bool = np.random.rand(1)
        if r_bool < chance:
            seq_r = np.flip(seq, 0).copy() # n x 5 shape
            features_r = [np.flip(item, 0).copy() for item in features] # n
            mat_r = np.flip(mat, [0, 1]).copy() # n x n

            # Complementary sequence
            seq_r = self.complement(seq_r)
        else:
            seq_r = seq
            features_r = features
            mat_r = mat
        return seq_r, features_r, mat_r

    def complement(self, seq, chance = 0.5):
        '''
        Complimentary sequence
        '''
        r_bool = np.random.rand(1)
        if r_bool < chance:
            seq_comp = np.concatenate([seq[:, 1:2],
                                       seq[:, 0:1],
                                       seq[:, 3:4],
                                       seq[:, 2:3],
                                       seq[:, 4:5]], axis = 1)
        else:
            seq_comp = seq
        return seq_comp

    def get_data_at_interval(self, start, end):
        '''
        Slice data from arrays with transformations
        '''
        # Sequence processing
        seq = self.seq.get(start, end)
        # Features processing
        features = [item.get(self.chr_name, start, end) for item in self.genomic_features]
        # Hi-C matrix processing
        mat = self.mat.get(start)
        mat = resize(mat, (self.image_scale, self.image_scale), preserve_range=True, anti_aliasing=True)
        mat = np.log(mat + 1)
        return seq, features, mat

    def get_active_intervals(self):
        '''
        Get intervals for sample data: [[start, end]]
        '''
        chr_bins = len(self.seq) / self.res
        data_size = (chr_bins - self.sample_bins) / self.stride
        starts = np.arange(0, data_size).reshape(-1, 1) * self.stride
        intervals_bin = np.append(starts, starts + self.sample_bins, axis=1)
        intervals = intervals_bin * self.res
        return intervals.astype(int)

    def filter(self, intervals, omit_regions):
        valid_intervals = []
        for start, end in intervals: 
            # Way smaller than omit or way larger than omit
            start_cond = start <= omit_regions[:, 1]
            end_cond = omit_regions[:, 0] <= end
            #import pdb; pdb.set_trace()
            if sum(start_cond * end_cond) == 0:
                valid_intervals.append([start, end])
        return valid_intervals

    def encode_seq(self, seq):
        ''' 
        encode dna to onehot (n x 5)
        '''
        seq_emb = np.zeros((len(seq), 5))
        seq_emb[np.arange(len(seq)), seq] = 1
        return seq_emb

    def shift_aug(self, target_size, start, end):
        '''
        All unit are in basepairs
        '''
        offset = random.choice(range(end - start - target_size))
        return start + offset , start + offset + target_size

    def shift_fix(self, target_size, start, end):
        offset = 0
        return start + offset , start + offset + target_size

    def check_length(self):
        assert len(self.seq.seq) == self.genomic_features[0].length(self.chr_name), f'Sequence {len(self.seq)} and First feature {self.genomic_features[0].length(self.chr_name)} have different length.' 
        assert abs(len(self.seq) / self.res -  len(self.mat)) < 2, f'Sequence {len(self.seq) / self.res} and Hi-C {len(self.mat)} have different length.' 

def get_feature_list(root_dir, feat_dicts):

    feat_list = []
    for feat_item in feat_dicts:
        file_name = feat_item['file_name']
        file_path = f'{root_dir}/{file_name}'
        norm = feat_item['norm']
        feat_list.append(data_feature.GenomicFeature(file_path, norm))
    return feat_list

def proc_centrotelo(bed_dir):
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