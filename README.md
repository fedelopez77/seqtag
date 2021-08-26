# seqtag
Sequence tagging with transformer models.

## How to run this module
### 0. Install dependencies
```
pytorch >= 1.7
transformers
sklearn
pandas
tensorboardx
```

### 1. Setup
Create a folder `data` and drop the files `train_atis.txt`, `dev_atis.txt` and `test_atis.txt` there.
To create the remaining necessary folders, just run `sh init.sh`

### 2. Preprocess
Preprocess the data by running:
```
python preprocess.py --run_id=atis --language_model=bert-base-uncased
```
The file with the preprocessed data will be stored in the folder `data/{run_id}_{language_model}`.
The available language models are in [Available Models](#available-models)

### 3. Train
To train you need to use `torch.distributed.launch` which allows for distributed training
over multiple CPU cores or GPUs.
```
python -m torch.distributed.launch --master_port=3001 --nproc_per_node=1 train.py \\
    --data=data/atis_bert-base-uncased.pt \\
    --run_id=foo-1 \\ 
    --n_procs=1 \\ 
    --epochs=100
```

### 4. Inference
Perform inference on the test set:
```
python -m torch.distributed.launch --master_port=3001 --nproc_per_node=1 infer.py \\
    --file=data/test_atis.txt \\
    --load_model=ckpt/foo-1-best-68ep
```
The output file will be written to `data/test_atis.txt.out` in this example.

### 5. Evaluate with CoNLL script
```
paste  data/test_atis.txt data/test_atis.txt.out | awk '{print  $1, $2, $4}' > res.txt
python2 conlleval.py res.txt
```


## Available models:
```
bert-base-uncased
xlm-mlm-en-2048
roberta-base
distilbert-base-uncased
facebook/bart-base
microsoft/layoutlm-base-uncased
```
