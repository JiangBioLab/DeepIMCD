import pickle
from typing import List,Dict
from collections import OrderedDict
import numpy as np
import pandas as pd
import torch
import tqdm
from torch import nn
import os
import h5py
from esm.models.esmc import ESMC
os.environ["INFRA_PROVIDER"] = "True"

def batch_sequences(sequences, batch_size=64):
    for i in range(0, len(sequences), batch_size):
        yield sequences[i:i + batch_size]

def masked_mean_pooling(embs, input_ids, pad_token_id):
    mask = (input_ids != pad_token_id).to(embs.dtype)
    mask = mask.unsqueeze(-1)
    summed = (embs * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts
    
def generate_embeddings(model, sequences, device, batch_size=64):
    embeddings = []
    pad_token_id = model.tokenizer.pad_token_id
    for batch in batch_sequences(sequences, batch_size):
        with torch.no_grad(), torch.autocast(enabled=True, device_type="cuda", dtype=torch.bfloat16):
            input_ids = model._tokenize(batch)
            output = model(input_ids.to(device))
            embs = output.embeddings
            batch_embeddings = masked_mean_pooling(embs,input_ids,pad_token_id)
            batch_embeddings = batch_embeddings.to(torch.float32).cpu().detach().numpy()
            embeddings.append(batch_embeddings)
    return np.vstack(embeddings) if embeddings else np.array([])

class HDF5_FILE():
    def __init__(self, data_dir: str, hdf5_path: str,sequence_column: str = 'aaSeqCDR3',column_seq: str = '\t',
                 sequence_clonefraction_column: str = 'cloneCount',v_column:str='V',
                 sequence_characters: tuple =
                 ('A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y')
                 , h5py_dict: dict = None,device: torch.device = torch.device('cuda:0')):
        self.data_dir = data_dir
        self.hdf5_path = hdf5_path
        self.sequence_column = sequence_column
        self.v_column=v_column
        self.sequence_cloneFraction_column = sequence_clonefraction_column
        self.column_seq = column_seq
        self.aas = sequence_characters
        self.n_aa = len(self.aas)
        self.aa_ind_dict = OrderedDict(zip(self.aas, range(self.n_aa)))
        self.h5py_dict = h5py_dict if h5py_dict is not None else dict(compression='gzip', compression_opts=6,
                                                                      chunks=True, shuffle=True)
        self.data_files = [self.data_dir]
        self.sample_keys = [os.path.basename(data_file).rsplit('.', 1)[0] for data_file in self.data_files]
        self.n_samples = len(self.sample_keys)
        with open(os.path.join('MIL_model','configs','tcr_score_chi2_score.pkl'), 'rb') as f:
            self.chi2_score = pickle.load(f)
        with open(os.path.join('MIL_model','configs','gene_dir.pkl'), 'rb') as f:
            self.gene_dir = pickle.load(f)
        self.device = torch.device(device)
        self.model = ESMC.from_pretrained("esmc_300m",device=self.device)
        self.model = self.model.to(self.device)
        self.model.eval()
        
    
    def read_sequence(self, filename: str):
        try:
            data = pd.read_csv(filename, header=0, sep=self.column_seq)
            data = data.sort_values(by=self.sequence_cloneFraction_column,ascending=False)
            data = data[[len(seq) <= 18 for seq in data[self.sequence_column].values]]
            data = data[[len(seq) >= 11 for seq in data[self.sequence_column].values]]
            data['score'] = [self.chi2_score.get(i,0) for i in data[self.sequence_column].values]
            data = data.sort_values(by='score', ascending=False)
            data = data.iloc[:8000,:]
            v_gene = np.array([self.gene_dir['v'][g.split('*')[0]] for g in data[self.v_column]], dtype=np.int8)
            embeddings = generate_embeddings(self.model, data[self.sequence_column].tolist(), self.device, 128)
            n_seq = len(data)
            return embeddings,v_gene,n_seq
        except Exception as e:
            print(f"Fail in {filename}")
            raise e
    
    def save_one_file(self,item):
        file_path,file_name = item
        if not os.path.exists(os.path.join(self.hdf5_path,file_name+'.hdf5')):
            embeddings,v_gene,n_seq = self.read_sequence(file_path)
            with h5py.File(os.path.join(self.hdf5_path,file_name+'.hdf5'), mode="w") as hf:
                group_ = hf.create_group('sampledata')
                group_.create_dataset('embeddings', data=embeddings, dtype=np.float64, **self.h5py_dict)
                group_.create_dataset('v_gene', data=v_gene, dtype=np.int8, **self.h5py_dict)
                group_.create_dataset('n_seqs', data=n_seq, dtype=np.int64)
        
    
    def save_hdf5_file(self):
        item_list = list(zip(self.data_files,self.sample_keys))
        for item in tqdm.tqdm(item_list,total=len(item_list)):
            self.save_one_file(item)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim_mult=4, dropout=0.1):
        super().__init__()
        hidden_dim = dim * hidden_dim_mult
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout))
    def forward(self, x):
        return self.net(x)

class LinearFeatureExtractor(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.2):
        super().__init__()
        self.hidden_dim = output_dim
        self.feat_proj_linear = nn.Linear(input_dim, output_dim)
        self.feat_proj_norm = nn.BatchNorm1d(output_dim)
        self.feat_proj_act = nn.GELU()
        self.feat_proj_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, K, _ = x.shape
        if K == 0: return torch.zeros(B, 0, self.hidden_dim, device=x.device)
        x = self.feat_proj_linear(x)
        x_reshaped = x.reshape(B * K, self.hidden_dim)
        x_bn = self.feat_proj_norm(x_reshaped)
        x = x_bn.reshape(B, K, self.hidden_dim)
        x = self.feat_proj_act(x)
        x = self.feat_proj_dropout(x)
        return x

class CustomAttentionPooling(nn.Module):
    def __init__(self, input_dim, num_pooling_queries=4, num_heads=2, dropout=0.1):
        super().__init__()
        self.num_pooling_queries = num_pooling_queries; self.input_dim = input_dim; self.num_heads = num_heads
        self.cls_token_global = nn.Parameter(torch.zeros(1, 1, input_dim)); self.other_global_queries = nn.Parameter(torch.zeros(1, self.num_pooling_queries, input_dim))
        nn.init.xavier_uniform_(self.cls_token_global); nn.init.xavier_uniform_(self.other_global_queries)
        self.internal_norm = nn.LayerNorm(input_dim)
        self.internal_self_attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=self.num_heads, dropout=dropout, batch_first=True)
        self.internal_dropout = nn.Dropout(dropout)
        self.cross_norm_global = nn.LayerNorm(input_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=self.num_heads, dropout=dropout, batch_first=True)
        self.cross_dropout = nn.Dropout(dropout)
        
    def forward(self, features, key_padding_mask=None):
        batch_size = features.size(0)
        cls_expanded = self.cls_token_global.expand(batch_size, -1, -1)
        queries_expanded = self.other_global_queries.expand(batch_size, -1, -1)
        global_tokens_g = torch.cat((cls_expanded, queries_expanded), dim=1)
        g_normed = self.internal_norm(global_tokens_g)
        g2_attn_output, _ = self.internal_self_attn(query=g_normed, key=g_normed, value=g_normed, need_weights=False)
        global_tokens_g2 = global_tokens_g + self.internal_dropout(g2_attn_output)
        if features.size(1) == 0:
            aggregated_cls_feature_from_g2 = global_tokens_g2[:, 0]
            return aggregated_cls_feature_from_g2
        g2_for_cross_normed = self.cross_norm_global(global_tokens_g2)
        attn_output_cross, _ = self.cross_attn(query=g2_for_cross_normed, key=features, value=features, key_padding_mask=key_padding_mask, need_weights=False)
        aggregated_cls_feature = attn_output_cross[:, 0]
        return self.cross_dropout(aggregated_cls_feature)


class My_Model2(nn.Module):
    def __init__(self, n_classes=5,sequence_reduction_fraction=1.0,embedding=True,max_len=8000,
                 embedding_dim=960,hidden_dim=128,dropout=[0.2,0.15,0.3],num_gene_types=64,num_transformer_layers=1,num_heads=2,
                 num_pooling_queries=3,
                 device: torch.device = torch.device('cuda:0')):
        super(My_Model2, self).__init__()
        self.device = device
        self.n_classes = n_classes
        self.sequence_reduction_fraction = sequence_reduction_fraction
        self.embedding=embedding
        self.max_len=max_len
        self.embedding_dim=embedding_dim
        self.hidden_dim=hidden_dim
        self.num_gene_types=num_gene_types
        self.num_transformer_layers=num_transformer_layers
        self.num_heads=num_heads
        self.num_pooling_queries=num_pooling_queries
        self.dropout=dropout
        self.feature_extractor = LinearFeatureExtractor(self.embedding_dim, self.hidden_dim, dropout=self.dropout[0])
        self.film_gamma_generator_gene = nn.Embedding(self.num_gene_types, self.hidden_dim)
        self.film_beta_generator_gene = nn.Embedding(self.num_gene_types, self.hidden_dim)
        
        self.transformer_layers = nn.ModuleList()
        for _ in range(self.num_transformer_layers):
            self.transformer_layers.append(nn.ModuleDict({
                'norm_sa': nn.LayerNorm(self.hidden_dim),
                'self_attn': nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=self.num_heads, dropout=self.dropout[1], batch_first=True),
                'dropout_sa': nn.Dropout(self.dropout[1]),
                'norm_ffn': nn.LayerNorm(self.hidden_dim),
                'ffn': FeedForward(self.hidden_dim, dropout=self.dropout[1]),
                'dropout_ffn': nn.Dropout(self.dropout[1])
            }))
        self.norm_pool = nn.LayerNorm(self.hidden_dim)
        self.pooling_module = CustomAttentionPooling(
                input_dim=self.hidden_dim,
                num_pooling_queries=self.num_pooling_queries,
                num_heads=1,
                dropout=dropout[1]
            )
        self.class_head = nn.Sequential(
            nn.BatchNorm1d(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout[2]),
            nn.Linear(self.hidden_dim // 2 , self.n_classes)
        )
    
    def pad_by_sample(self,data, n_sample, v_gene):
        B = n_sample.numel()
        feat_dim = data.size(1)
        data_pad = torch.zeros(B, self.max_len, feat_dim,dtype=data.dtype,device=data.device)
        v_gene_pad = torch.zeros(B, self.max_len,dtype=v_gene.dtype,device=v_gene.device)
        start = 0
        for i, n in enumerate(n_sample.tolist()):
            end = start + n
            data_pad[i, :n] = data[start:end]
            v_gene_pad[i, :n] = v_gene[start:end]
            start = end
        return data_pad, v_gene_pad
    
    def compute_dpp_loss(self):
        queries = self.pooling_module.other_global_queries.squeeze(0)
        norm_queries = F.normalize(queries, p=2, dim=1)
        sim_matrix = torch.matmul(norm_queries, norm_queries.t())
        identity = torch.eye(sim_matrix.size(0), device=sim_matrix.device)
        return F.mse_loss(sim_matrix, identity)
        
        
    def forward(self, data, n_sample,v_gene):
        data,v_gene = self.pad_by_sample(data, n_sample,v_gene)
        B, K_orig, _ = data.shape
        x = self.feature_extractor(data)
        gamma = self.film_gamma_generator_gene(v_gene)
        beta = self.film_beta_generator_gene(v_gene)
        x = gamma * x + beta
        final_x_for_transformer = x
        processed_seqs = final_x_for_transformer
        for layer_module in self.transformer_layers:
            sa_input = layer_module['norm_sa'](processed_seqs)
            attn_output, _ = layer_module['self_attn'](query=sa_input, key=sa_input, value=sa_input, key_padding_mask=None, need_weights=False)
            processed_seqs = processed_seqs + layer_module['dropout_sa'](attn_output)
            ffn_input = layer_module['norm_ffn'](processed_seqs)
            ffn_output = layer_module['ffn'](ffn_input)
            processed_seqs = processed_seqs + layer_module['dropout_ffn'](ffn_output)
        pool_input_normed = self.norm_pool(processed_seqs)
        aggregated_features = self.pooling_module(pool_input_normed, key_padding_mask=None)
        logits = self.class_head(aggregated_features)
        return logits

    def reduce_and_stack_minibatch(self, targets, sequences_embedding,v_genes):
        with (torch.no_grad()):
            sequences_embedding = [t.to(self.device) for t in sequences_embedding]
            v_gene = [t.to(self.device) for t in v_genes]
            input_list = sequences_embedding
            reduced_inputs, reduced_v = list(
                zip(*[self.reduce_sequences_for_bag(inp,v_g) for inp,v_g in zip(input_list,v_gene)]))
            mb_targets = torch.stack(targets, dim=0).to(device=self.device)
            mb_reduced_v = torch.cat(reduced_v, dim=0)
            mb_reduced_inputs = torch.cat(reduced_inputs, dim=0)
            mb_n_sequences = torch.tensor([len(i) for i in reduced_v], dtype=torch.long,
                                          device=self.device)
        return mb_targets, mb_reduced_inputs,mb_reduced_v, mb_n_sequences


    def reduce_sequences_for_bag(self, inputs,v_genes):
        with torch.no_grad():
            reduced_inputs = inputs.detach().to(self.device, dtype=torch.float)
            reduced_v_genes = v_genes.detach().to(self.device, dtype=torch.long)
        return reduced_inputs, reduced_v_genes

def get_feature(file_path: str):
    with h5py.File(os.path.join(file_path), mode="r") as hf:
        squences_embedding = hf['sampledata']['embeddings'][:]
        v_genes = hf['sampledata']['v_gene'][:]
    return squences_embedding,v_genes


def get_mil_result(file_path: str,device='cpu'):
    file_name = os.path.basename(file_path).rsplit('.', 1)[0]
    tmp_path = os.path.join('MIL_model','tmp')
    hdf5_file = HDF5_FILE(data_dir=os.path.join(file_path), hdf5_path=tmp_path,device=device)
    hdf5_file.save_hdf5_file()
    model = My_Model2()
    model_path = os.path.join('MIL_model','configs','model_mil.pkl')
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.device=device
    model.to(device)
    model.eval()
    squences_embedding,v_genes = get_feature(os.path.join(tmp_path,file_name+'.hdf5'))
    with torch.no_grad():
        labels = torch.tensor([0,0,0,0,0])
        squences_embedding,v_genes=torch.tensor(squences_embedding),torch.tensor(v_genes)
        labels, inputs, v_genes, n_sequences = model.reduce_and_stack_minibatch([labels], [squences_embedding],[v_genes])
        result = model(data=inputs, n_sample=n_sequences,v_gene=v_genes)
        result = torch.softmax(result, dim=1).cpu().numpy().reshape(-1)
    for filename in os.listdir(os.path.join(tmp_path)):
        os.remove(os.path.join(tmp_path,filename))
    return result
