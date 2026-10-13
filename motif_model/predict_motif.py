import pickle
from typing import List
import pandas as pd
import torch
import tqdm
from torch import nn
import os


def cut_kmer(seq: str):
    kmer_list = []
    for k in range(3, 7 + 1):
        kmer_list.extend([seq[i:i + k] for i in range(len(seq) - k + 1)])
    return kmer_list


def trans_data(feature: List[float], k1: float = 1e-6):
    with open(os.path.join('motif_model','configs','t_l.pkl'), 'rb') as f:
        t_l = pickle.load(f)
    feature = torch.Tensor(feature)
    feature = torch.log(feature + k1)
    feature = (feature - t_l[0]) / t_l[1]
    feature = (feature - t_l[2]) / (t_l[3] - t_l[2])
    return feature


def get_feature(file_path: str):
    with open(os.path.join('motif_model','configs','kmer_3_7_500000.pkl'), 'rb') as f:
        kmer_list = pickle.load(f)[:180000]
    kmer_dict = dict(zip(kmer_list, range(len(kmer_list))))

    data = pd.read_csv(file_path, sep='\t', header=0)
    data = data[[len(seq) <= 18 for seq in data['aaSeqCDR3'].values]]
    data = data[[len(seq) >= 11 for seq in data['aaSeqCDR3'].values]]
    seq_list = data.iloc[:, 0].tolist()
    count_list = data.iloc[:, -1].tolist()
    feature = [0] * 180000
    for s, c in tqdm.tqdm(zip(seq_list, count_list), total=len(seq_list)):
        k_dict = cut_kmer(s[1:-1])
        for k in k_dict:
            if k in kmer_dict:
                feature[kmer_dict[k]] += c
    sum_count = sum(count_list)
    feature = [i / sum_count for i in feature]
    feature = trans_data(feature)
    return feature


class Kmer_Model1(nn.Module):
    def __init__(self, in_feature: int, out_feature: int):
        super(Kmer_Model1, self).__init__()
        self.out1 = nn.Linear(in_feature, out_features=out_feature)

    def forward(self, input):
        output = self.out1(input)
        return output


def get_motif_result(file_path: str, device='cpu'):
    feature = get_feature(file_path).to(device)
    model = Kmer_Model1(in_feature=180000, out_feature=5).to(device)
    model.load_state_dict(torch.load(os.path.join('motif_model','configs','model_motif.pt'), map_location=device, weights_only=True))
    with torch.no_grad():
        result = model(feature)
        result = torch.softmax(result, dim=1).cpu().numpy().reshape(-1)
    return result
