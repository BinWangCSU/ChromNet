import torch
import torch.nn as nn
from HierarchicalFeatureModule import PatchSampledModel
from ContextualEncodingModule import SequenceGenerator
from ChromatinInteractionReconstructor import Decoder

class ChromNet(nn.Module):
    def __init__(self, num_genomic_features, num_cell_types):
        super(ChromNet, self).__init__()
        print('Initializing ChromNet')
        self.encoder = PatchSampledModel(input_channels=num_genomic_features, output_channels=16, patch_length_1=10, patch_length_2=256, topk=10, kernel_size=3, stride=1)
        self.tranfer = SequenceGenerator(seq_dim=128, hidden_dim=128 * 2)
        self.decoder = Decoder(128 * 2, hidden=128, num_blocks=1)

        self.conv_classifier = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64 * 64 * 64, 256),
            nn.ReLU(),
            nn.Linear(256, num_cell_types)
        )

    def move_feature_forward(self, x):
        return x.transpose(1, 2).contiguous()
    
    def forward(self, x):
        x = self.move_feature_forward(x).float()
        x = self.encoder(x)
        x = self.move_feature_forward(x)
        x1 = self.tranfer(x)
        classification_output=self.conv_classifier(x1)
        x = self.decoder(x1).squeeze(1)
        
        return x, classification_output

if __name__ == '__main__':
    main()
