import os
import math
import sqlite3
import numpy as np
import pandas as pd
import argparse
import faiss
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from cluster_model.model.classification_model import MLP
from cluster_model.model.transformer_encoder import TransformerModel
from cluster_model.model.dataset import TCRDataset

from multiprocessing import Manager, Process

def faiss_block(xb,xq,nb,nq,block_size,k):
    num_gpu = torch.cuda.device_count()
    d = xq.size(1)
    TMP_I = []
    block_id = 0
    for start in range(0, nb, block_size):
        end = min(start + block_size, nb)
        xb_block = xb[start:end]

        gpu_resources = []
        for i in range(num_gpu):
            res = faiss.StandardGpuResources()
            gpu_resources.append(res)
        gpu_index = faiss.IndexProxy()
        for i, res in enumerate(gpu_resources):
            cfg = faiss.GpuIndexFlatConfig()
            cfg.useFloat16 = False
            cfg.device = i
            shard_index = faiss.GpuIndexFlatL2(res, d, cfg)
            gpu_index.addIndex(shard_index)
        gpu_index.add(xb_block)
        D, I = gpu_index.search(xq, k)
        del gpu_index
        torch.cuda.empty_cache()
        ab_I = I + start
        ab_I = ab_I.flatten().tolist()
        TMP_I.extend(ab_I)
        block_id += 1

    gpu_resources = []
    for i in range(num_gpu):
        res = faiss.StandardGpuResources()
        gpu_resources.append(res)
    gpu_index = faiss.IndexProxy()
    for i, res in enumerate(gpu_resources):
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = i
        shard_index = faiss.GpuIndexFlatL2(res, d, cfg)
        gpu_index.addIndex(shard_index)

    TMP_index = np.array(list(set(TMP_I))).astype(int)
    TMP_xb = xb[TMP_index]
    gpu_index.add(TMP_xb)

    final_D, final_I = gpu_index.search(xq, k)
    del gpu_index
    torch.cuda.empty_cache()
    final_I_mapped = np.empty_like(final_I)
    for i in range(nq):
        final_I_mapped[i] = [TMP_index[orig_index] for orig_index in final_I[i]]
    return final_D, final_I_mapped

def make_query(dM_q, dM_r, thr=0.2, GPU=False):
    dM_q=dM_q.to(torch.float32)
    dM_r=dM_r.to(torch.float32)
    nq=dM_q.size(0)
    nr=dM_r.size(0)
    D, I = faiss_block(dM_r,dM_q,nr,nq,100000000,1)
    # print('Searching nearest neighbor.Done.')
    vv=np.where((D[:,0]<=thr))[0]
    q_index = vv
    r_index = I[vv, 0]
    return q_index, r_index


def EncodeCDR3s(CDR3s,model,batch_size,device):
    Encoded_CDR3s = []
    with torch.no_grad():
        model.eval()
        batchsize = batch_size
        cur = 0
        while True:
            if cur+batchsize<len(CDR3s):
                batch_cdr3s = CDR3s[cur:cur+batchsize]
                cur+=batchsize
                inputs = amino_acid_to_index(batch_cdr3s)
                inputs = inputs.to(device)
                output = model(inputs)
                output = F.normalize(output, dim=1)
                Encoded_CDR3s.append(output.detach().cpu())
            else:
                batch_cdr3s = CDR3s[cur:len(CDR3s)]
                inputs = amino_acid_to_index(batch_cdr3s)
                inputs = inputs.to(device)
                output = model(inputs)
                output = F.normalize(output, dim=1)
                Encoded_CDR3s.append(output.detach().cpu())
                break
    Encoded_CDR3s = torch.cat(Encoded_CDR3s,dim=0)
    return Encoded_CDR3s

def amino_acid_to_index(sequences):
    amino_acids = "ARNDCEQGHILKMFPSTWYV"
    aa_to_idx = {aa: idx+1 for idx, aa in enumerate(amino_acids)}
    batch_indices = []
    for sequence in sequences:
        indices = [aa_to_idx[aa] for aa in sequence[3:-2]]
        if len(sequence[3:-2])<15:
            for i in range(len(sequence[3:-2]),15):
                indices.append(0)
        batch_indices.append(indices)
    return torch.LongTensor(batch_indices)

def Frequency(df):
    disease_category = ["Breast", "Colorectal", "Healthy", "Liver", "Lung"]
    franc = np.zeros((1, len(disease_category)))
    qdf_ex = df[df[6]!='query']
    category_dfs = {category: qdf_ex[qdf_ex[5] == category] for category in disease_category}
    for j in range(len(disease_category)):
        ddf = category_dfs[disease_category[j]]
        franc[0][j] = len(ddf)
    row_sums = franc.sum(axis=1, keepdims=True)
    franc_normalized = franc / (row_sums+1e-6)
    return torch.Tensor(franc_normalized)

def get_cluster_result(file_path: str, device='cpu'):
    parser = argparse.ArgumentParser(description='TCR Contrastive Learning')
    batch_size = 40000
    d_model = 256
    nhead = 8
    num_layers = 6
    out_dim = 64
    model = TransformerModel(21, d_model, nhead, num_layers, out_dim)
    model = nn.DataParallel(model)
    if os.path.exists('cluster_model/weights/TCREncoder/checkpoint_0020.pth.tar'):
        state = torch.load('cluster_model/weights/TCREncoder/checkpoint_0020.pth.tar', map_location=device)
        model.load_state_dict(state['state_dict'])
        model = model.to(device)
    else:
        print('There is no TCREncoder checkpoint.Exit!')
        exit(0)
    conn = sqlite3.connect('cluster_model/db/multi_cancer_reference_speTCR.db')

    ref_df = pd.read_sql_query("SELECT * FROM SpecificTCR;",conn)
    ref_trbvs = ref_df['TRBV'].unique()
    ref_Encoded_CDR3s={}
    if os.path.isdir(file_path):
        qFs=os.listdir(file_path)
        qFileList=[]
        for ff in qFs:
            qFileList.append(os.path.join(file_path,ff))
    elif os.path.isfile(file_path):
        qFileList=[file_path]
    else:
        print("file is not exists.")

    for qFile in qFileList:
        q_dir = qFile.rstrip(qFile.split('/')[-1])
        q_name = qFile.split('/')[-1].rstrip('.txt')
        q_dataset = TCRDataset(qFile)

        q_trbv_cdr3 = {}
        for uid in q_dataset.uids:
            trbv = q_dataset.trbvs[uid]
            cdr3 = q_dataset.tcrs[uid]
            if trbv not in q_trbv_cdr3:
                q_trbv_cdr3[trbv]=[cdr3]
            else:
                q_trbv_cdr3[trbv].append(cdr3)

        cluster_id = 0
        records = []
        for trbv in q_trbv_cdr3.keys():
            if trbv not in ref_trbvs:
                continue
            df = ref_df[ref_df['TRBV']==trbv]
            rIDs = df['ID'].values
            rCDR3s = df['CDR3'].values
            rTRBVs = df['TRBV'].values
            qCDR3s = q_trbv_cdr3[trbv]

            if trbv in ref_Encoded_CDR3s:
                rEncoded_CDR3s = ref_Encoded_CDR3s[trbv]
            else:
                rEncoded_CDR3s = EncodeCDR3s(rCDR3s, model,batch_size,device)
                ref_Encoded_CDR3s[trbv]=rEncoded_CDR3s
            qEncoded_CDR3s = EncodeCDR3s(qCDR3s, model,batch_size,device)
            q_index, r_index = make_query(qEncoded_CDR3s, rEncoded_CDR3s, GPU=torch.cuda.is_available())
            for idx1, idx2 in zip(q_index, r_index):
                cluster_id += 1
                cdr3 = qCDR3s[idx1]
                uid = q_dataset.unique_dict[cdr3 + '_' + trbv]

                records.append([
                    q_dataset.tcrs[uid],
                    q_dataset.trbvs[uid],
                    cluster_id,
                    0.0,
                    '',
                    '',
                    'query',
                ])
                if uid in q_dataset.link_dict:
                    for linkid in q_dataset.link_dict[uid]:
                        records.append([
                            q_dataset.tcrs[linkid],
                            q_dataset.trbvs[linkid],
                            cluster_id,
                            0.0,
                            '',
                            '',
                            'query'
                        ])
                tr_index = int(rIDs[idx2])
                row = df[df['ID'] == tr_index].iloc[0]
                records.append([
                    row['CDR3'],
                    row['TRBV'],
                    cluster_id,
                    row['Freq'],
                    row['SampleID'],
                    row['Type'],
                    'ref',
                ])
        result_df = pd.DataFrame(records)
        input_tensor = Frequency(result_df)
        # Load best classification model for test evaluation
        classify_model_state = torch.load('cluster_model/weights/CancerClassification/mlp_best_pytorch.pth', map_location=device)
        classify_model = MLP(
            hidden_sizes=(256,64),
            activation='tanh',
            init_type='he'
        ).to(device)
        classify_model.load_state_dict(classify_model_state)
        classify_model.eval()
        # Test predictions
        with torch.no_grad():
            outputs = classify_model(input_tensor.to(device))
            prob = torch.softmax(outputs, dim=1).cpu().numpy()
            _, pred = torch.max(outputs, 1)
    conn.close()
    return prob[0]

