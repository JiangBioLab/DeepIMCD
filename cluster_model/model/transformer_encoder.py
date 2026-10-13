import torch
import torch.nn as nn


class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, out_dim):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(d_model, out_dim)
        self.d_model = d_model

    def forward(self, src):
        src_key_padding_mask = (src == 0)
        src = self.embedding(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        src = self.pos_encoder(src)
        src = src.transpose(0, 1)
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        output = output.transpose(0, 1)
        # Masking the padding tokens for mean calculation
        mask = ~src_key_padding_mask.unsqueeze(-1)
        masked_output = output * mask
        summed_output = masked_output.sum(dim=1)
        non_padding_count = mask.sum(dim=1)
        mean_output = summed_output / non_padding_count
        final_output = self.fc(mean_output)
        return final_output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x = x.transpose
        x = x + self.pe[:, :x.size(1), :]
        # x = x.transpose
        return self.dropout(x)