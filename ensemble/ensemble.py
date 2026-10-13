import numpy as np
import textwrap
from prettytable import PrettyTable


def get_ensemble_result(motif_result,cluster_result,mil_result):
    ensemble_result = (np.array(motif_result)+np.array(cluster_result)+np.array(mil_result))/3
    table = PrettyTable()
    table.field_names = ["model","Breast", "Colorectal", "Healthy", "Liver", "Lung"]
    table.add_row(['motif-based model']+[f"{float(x):.4f}" for x in motif_result.tolist()])
    table.add_row(['clustering-based model']+[f"{float(x):.4f}" for x in cluster_result.tolist()])
    table.add_row(['MIL-based model']+[f"{float(x):.4f}" for x in mil_result.tolist()])
    table.add_row(['DeepIMCD']+[f"{float(x):.4f}" for x in ensemble_result.tolist()])
    print(f"\t")
    table_str = textwrap.indent(str(table), "\t\t")
    print(f"{table_str}\n")