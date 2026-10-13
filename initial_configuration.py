import hashlib
import requests
from pathlib import Path
import argparse
import os
import tqdm

def get_md5(file_path):
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(8 * 1024 * 1024)
            if not chunk:
                break
            md5.update(chunk)
    return md5.hexdigest()
    
def download(url:str,save_path:Path,expected_md5:str):
    save_path.parent.mkdir(parents=True,exist_ok=True)
    if save_path.exists():
        actual_md5 = get_md5(save_path)
        print("File already exists.")
        print("Actual MD5 :", actual_md5)
        print("Expected MD5:", expected_md5)
        if actual_md5.lower() == expected_md5.lower():
            print("MD5 checksum passed.")
            return str(save_path)
        print("MD5 mismatch, downloading again.")
        save_path.unlink()

    with requests.get(url,stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        chunk_size = 8 * 1024 * 1024
        with open(save_path, "wb") as f:
            with tqdm.tqdm(total=total_size,unit="B",unit_scale=True,unit_divisor=1024,desc="Downloading") as progress_bar:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        progress_bar.update(len(chunk))
    actual_md5 = get_md5(save_path)
    print("Actual MD5 :", actual_md5)
    print("Expected MD5:", expected_md5)
    assert (actual_md5.lower()== expected_md5.lower()), "MD5 checksum failed"
    print("MD5 checksum passed.")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-action','--action', type=str, help='')
    args = parser.parse_args()
    if args.action=='download':
        download_list = [{'url':"https://zenodo.org/records/23012456/files/kmer_3_7_500000.pkl?download=1",
                          'save_path':Path("motif_model/configs/kmer_3_7_500000.pkl"),
                          'expected_md5':"15fb030bda25ebdbbceb0515fa5d6fde"},
                          {'url':"https://zenodo.org/records/23012456/files/model_motif.pt?download=1",
                          'save_path':Path("motif_model/configs/model_motif.pt"),
                          'expected_md5':"42be03246c99ef3f13e3bce72f6dd2ac"},
                          {'url':"https://zenodo.org/records/23012456/files/t_l.pkl?download=1",
                          'save_path':Path("motif_model/configs/t_l.pkl"),
                          'expected_md5':"482b78d3d8cdae5700653cdd6ee45658"},
                          {'url':"https://zenodo.org/records/23012456/files/model_mil.pkl?download=1",
                          'save_path':Path("MIL_model/configs/model_mil.pkl"),
                          'expected_md5':"46fdf4a515bb7bf21465f3889ed3d4de"},
                          {'url':"https://zenodo.org/records/23012456/files/gene_dir.pkl?download=1",
                          'save_path':Path("MIL_model/configs/gene_dir.pkl"),
                          'expected_md5':"21c9ec5b96c7e5d60f71a7dd2af66de7"},
                          {'url':"https://zenodo.org/records/23012456/files/tcr_score_chi2_score.pkl?download=1",
                          'save_path':Path("MIL_model/configs/tcr_score_chi2_score.pkl"),
                          'expected_md5':"b92a02376d376de3256835e22e64cb8d"},
                          {'url':"https://zenodo.org/records/23012456/files/esmc_300m_2024_12_v0.pth?download=1",
                          'save_path':Path("data/weights/esmc_300m_2024_12_v0.pth"),
                          'expected_md5':"000cdd4cb3b8e3e4391a884161fa7434"},
                          {'url':"https://zenodo.org/records/23012456/files/multi_cancer_reference_speTCR.db?download=1",
                          'save_path':Path("cluster_model/db/multi_cancer_reference_speTCR.db"),
                          'expected_md5':"cd8009c31d58b24962db04a13a825205"},
                          {'url':"https://zenodo.org/records/23012456/files/mlp_best_pytorch.pth?download=1",
                          'save_path':Path("cluster_model/weights/CancerClassification/mlp_best_pytorch.pth"),
                          'expected_md5':"0e548a393029fc3e5c5022e0cd4db44e"},
                          {'url':"https://zenodo.org/records/23012456/files/checkpoint_0020.pth.tar?download=1",
                          'save_path':Path("cluster_model/weights/TCREncoder/checkpoint_0020.pth.tar"),
                          'expected_md5':"f3da0188a77291fcd0436f6ab5bc6fee"}]
        for dl in download_list:
            download(dl['url'],dl['save_path'],dl['expected_md5'])