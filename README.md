HSHR
===========
Hypergraph-Guided Self-Supervised Hashing-Encoding Retrieval

Method and codes of the paper: Hypergraph-Guided Slide-Level Histology Retrieval with Self-Supervised Hashing.

## Requirement
- Python (3.7.0)
    - Openslide-python (1.1.1)
    - Pytorch (1.8.1)
    - scikit-image (0.16.2)
    - scikit-learn (1.0.2)


## Usage
THe following steps show how to run HSHR in your device.
### Step 1: Data Organization
Make the `./WSI` folder, download whole slide images there, and simply organize them into the following structure.
```bash
WSI
└── SUBTYPE_1
    └── TCGA-XXXX-PID01-XXXX-XXXX.svs
        TCGA-XXXX-PID01-XXXX-XXXX.svs
        TCGA-XXXX-PID02-XXXX-XXXX.svs
        TCGA-XXXX-PID03-XXXX-XXXX.svs
    SUBTYPE_2
    └── TCGA-XXXX-PID04-XXXX-XXXX.svs
        TCGA-XXXX-PID04-XXXX-XXXX.svs
        TCGA-XXXX-PID05-XXXX-XXXX.svs
        TCGA-XXXX-PID05-XXXX-XXXX.svs
    SUBTYPE_3
    └── TCGA-XXXX-PID06-XXXX-XXXX.svs
        TCGA-XXXX-PID07-XXXX-XXXX.svs
        TCGA-XXXX-PID08-XXXX-XXXX.svs
```
### Step 2: Preprocessing
To preprocess the raw WSIs, you need to specify the following arguments:
- SVS_DIR: The path of your WSI datasets.
- RESULT_DIR: A path to save your preprocessed results.
- TMP: The path to save some necessary tmp files.
```
python preprocess.py --SVS_DIR THE/PATH/OF/YOUR/WSI/DATASETS --RESULT_DIR A/PATH/TO/SAVE/YOUR/PREPROCESSED/RESULTS --TMP THE/PATH/TO/SAVE/SOME/NECESSARY/TMP/FILES
```

### Step 3: SSL Encoder Training
To train your SSL hash encoder, you need to specify the following arguments:
- RESULT_DIR = A path to save your preprocessed results.
- TMP = The path to save some necessary tmp files.
- MODEL_DIR = A path to save your trained hash encoder model.
- DATASETS = a list of datasets to test the performance of retrieval


```
python ssl_encoder_training.py --RESULT_DIR A/PATH/TO/SAVE/YOUR/PREPROCESSED/RESULTS --TMP THE/PATH/TO/SAVE/SOME/NECESSARY/TMP/FILES --MODEL_DIR = A/PATH/TO/SAVE/YOUR/TRAINED/HASH/ENCODER/MODEL --DATASETS SUBTYPE_A SUBTYPE_B SUBTYPE_C
```


### Step 4: Hypergraph-guided Retrieval
To retrieve with HSHR, you need to specify the following arguments:
- MODEL_DIR = A path to save your trained hash encoder model.
- DATASETS = a list of datasets to test the performance of retrieval

```
python hypergraph_retrieval.py --MODEL_DIR = A/PATH/TO/SAVE/YOUR/TRAINED/HASH/ENCODER/MODEL --DATASETS SUBTYPE_A SUBTYPE_B SUBTYPE_C
```
