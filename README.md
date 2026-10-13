# DeepIMCD



DeepIMCD is an ensemble artificial intelligence model for multi-cancer detection by focusing on different aspects of immune repertoires: TCR motif analysis, TCR sequence clustering, and TCR repertoire characterization. 



## Requirements



The project was developed and tested in a Conda environment.



A environment configuration is provided in: `environment.yml`



Create the environment using:



```bash

conda env create -f environment.yml

conda activate DeepIMCD

```



## Model configuration files

Pretrained model weights are **not included directly** in this repository because of their large file size.

Please **download the required configuration files from Zenodo** separately and place them in corresponding directory.



You can also use the built-in DeepIMCD script `initial_configuration.py` to download the configuration file.

`initial_configuration.py` will automatically check the existing configuration files in the directory and verify their md5 checksums. If any configuration files are missing or the md5 check fails, it will download the configuration files from Zenodo to the specified directory.



```bash

python initial_configuration.py --action download

```





Please make sure that the configuration files downloaded from Zenodo follow the directory structure below:



```text

DeepIMCD/
├── data/
│   └── weights/
│       └── esmc_300m_2024_12_v0.pth
├── ensemble/
│   └── ensemble.py
├── motif_model/
│   ├── configs/
│   │   ├── kmer_3_7_500000.pkl
│   │   ├── model_motif.pt
│   │   └── t_l.pkl
│   └── predict_motif.py
├── cluster_model/
│   ├── db/
│   │   └── multi_cancer_reference_speTCR.db
│   ├── model/
│   │   ├── classification_model.py
│   │   ├── dataset.py
│   │   └── transformer_encoder.py
│   ├── weights/
│   │   ├── CancerClassification/
│   │   │   └── mlp_best_pytorch.pth
│   │   └── TCREncoder/
│   │       └── checkpoint_0020.pth.tar
│   └── predict_cluster.py
├── MIL_model/
│   ├── configs/
│   │   ├── gene_dir.pkl
│   │   ├── model_mil.pkl
│   │   └── tcr_score_chi2_score.pkl
│   ├── tmp/
│   └── predict_mil.py
├── environment.yml
├── predict.py
└── README.md

```



## Usage



DeepIMCD needs a GPU to run. Users need to upload the file path of the TCR repertoire with predictions and specify the device.



```bash

python predict.py -sample_path "PATH_OF_TCR_REPERTOIRE" --device cuda:0

```



Each sample to be predicted should be provided as a tab-delimited text file representing one TCR repertoire.

The input file **must contain a header row**, and the column names must be exactly:

```text

aaSeqCDR3	V	J	cloneCount

```

The columns must appear in the following order:

1. `aaSeqCDR3`
2. `V`
3. `J`
4. `cloneCount`

The delimiter between columns must be a tab character (\t). Other delimiters, such as commas or spaces, are not supported.

The four columns represent the CDR3 amino acid sequence, V gene, J gene, and clone count, respectively.

Please ensure that the header names, column order, and tab-delimited format are preserved exactly, as deviations from this format may cause input parsing errors.



DeepIMCD will run predictions from three base models separately, and then output the final ensemble. As shown below:



```text
Start predicting of TCR motif-based model:

100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 397689/397689 \[00:04<00:00, 99202.88it/s]

Start predicting of TCR clustering-based model:

Start predicting of TCR MIL-based model:

100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 \[00:07<00:00,  7.33s/it]

Start DeepIMCD ensemble：



               +------------------------+--------+------------+---------+--------+--------+

               |         model          | Breast | Colorectal | Healthy | Liver  |  Lung  |

               +------------------------+--------+------------+---------+--------+--------+

               |   motif-based model    | 0.9495 |   0.0227   |  0.0145 | 0.0010 | 0.0122 |

               | clustering-based model | 0.9399 |   0.0402   |  0.0013 | 0.0083 | 0.0102 |

               |    MIL-based model     | 0.9437 |   0.0493   |  0.0014 | 0.0017 | 0.0039 |

               |        DeepIMCD        | 0.9444 |   0.0374   |  0.0057 | 0.0037 | 0.0088 |

               +------------------------+--------+------------+---------+--------+--------+



```







