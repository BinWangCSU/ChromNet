import torch
import torch.nn as nn

class SequenceGenerator(nn.Module):
    def __init__(self, seq_dim, hidden_dim):
        super(SequenceGenerator, self).__init__()

        # LSTM层
        self.lstm = nn.LSTM(input_size=seq_dim, hidden_size=hidden_dim, batch_first=True)

        # 初始化LSTM权重
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, input_sequence):

        batch_size = input_sequence.size(0)
        hidden = (torch.zeros(1, batch_size, self.lstm.hidden_size).to(input_sequence.device),
                  torch.zeros(1, batch_size, self.lstm.hidden_size).to(input_sequence.device))

        lstm_output, _ = self.lstm(input_sequence, hidden)

        seq_encoding_elementwise_product = lstm_output.unsqueeze(2) * lstm_output.unsqueeze(1)
        new_tensor = seq_encoding_elementwise_product.permute(0, 3, 1, 2)
        return new_tensor