# -*- coding: gbk -*-

import argparse
import pandas as pd
import torch
import numpy as np
from cluster_model.predict_cluster import get_cluster_result
from motif_model.predict_motif import get_motif_result
from MIL_model.predict_mil import  get_mil_result
from ensemble.ensemble import get_ensemble_result
import warnings
warnings.filterwarnings("ignore")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='DeepIMCD predict')
    parser.add_argument('-sample_path','--sample_path', type=str, help='path to test data')
    parser.add_argument('--device', help='Default: "cpu".', type=str, default="cpu")
    args = parser.parse_args()
    args.device = torch.device(args.device)
    
    print("Start predicting of TCR motif-based model:")
    motif_result=get_motif_result(args.sample_path,device=args.device)
    
    print("Start predicting of TCR clustering-based model:")
    cluster_result = get_cluster_result(args.sample_path,device=args.device)
    
    print("Start predicting of TCR MIL-based model:")
    mil_result=get_mil_result(args.sample_path,device=args.device)
    
    print("Start DeepIMCD ensemble:")
    get_ensemble_result(motif_result,cluster_result,mil_result)
    