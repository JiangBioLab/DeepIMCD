from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle
import os
import pandas as pd
import torch
import numpy as np
import random
import pandas as pd

class TCRDataset(Dataset):
    def __init__(self, file_path):
        tcr_df = pd.read_csv(file_path, sep='\t', header=0)
        self.tcrs = []
        self.trbvs = []
        self.link_dict = {}
        unique_dict = {}
        uid = 0
        for _, line in tcr_df.iterrows():
            cdr3=line['aaSeqCDR3']
            if self.check(cdr3):
                self.tcrs.append(cdr3)
                trbv = line['V']
                self.trbvs.append(trbv)
                if cdr3+'_'+trbv not in unique_dict:
                    unique_dict[cdr3+'_'+trbv]=uid
                else:
                    if unique_dict[cdr3+'_'+trbv] not in self.link_dict:
                        self.link_dict[unique_dict[cdr3+'_'+trbv]]=[uid]
                    else:
                        self.link_dict[unique_dict[cdr3+'_'+trbv]].append(uid)
                uid+=1
        self.uids = list(unique_dict.values())
        self.unique_dict = unique_dict
        # print("relative_index:"+str(self.uids.index(spe_uid)))
    def check(self, cdr3):
        if len(cdr3)<10 or len(cdr3)>20:
            return False
        if not cdr3.startswith('C') or not cdr3.endswith('F'):
            return False
        AAstring='ACDEFGHIKLMNPQRSTVWY'
        AAstringList=list(AAstring)
        i = 0
        for aa in cdr3:
            if aa not in AAstringList:
                break
            else:
                i+=1
        if i<len(cdr3):
            return False
        return True

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        cdr3 = self.tcrs[self.uids[idx]]
        return cdr3[3:-2]