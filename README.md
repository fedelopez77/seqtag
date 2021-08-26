# seqtag
Named Entity Recognition on the ATIS dataset with transformers.

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

## Results
| Model      |  Ma-P |  Ma-R | Ma-F1 |  Mi-P |  Mi-R | Mi-F1 |  Acc  | Params |
|------------|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|--------|
| BERT       | 80.97 | 78.22 | 77.59 | 95.36 | 95.52 | 95.44 | 98.06 | 110M   |
| RoBERTa    | 79.57 | 79.67 | 78.62 | 95.19 | 95.52 | 95.36 | 97.99 | 124M   |
| DistilBERT | 79.84 | 78.85 | 77.49 | 94.89 | 94.82 | 94.85 | 97.71 | 66M    |
| BART       | 79.13 | 78.21 | 77.30 | 95.35 | 95.42 | 95.38 | 98.08 | 140M   |
| LayoutLM   | 79.12 | 79.37 | 77.66 | 95.15 | 95.42 | 95.28 | 98.05 | 113M   |

## Available models:
The results reported here were obtained with the following models and checkpoints. 
However, more architectures and all corresponding checkpoints can be run as well.
See [Hugging Face pretrained models](https://huggingface.co/transformers/pretrained_models.html).

| Architecture | Checkpoint ID                         |
|--------------|---------------------------------------|
| BERT         | ```bert-base-uncased```               |
| RoBERTa      | ```roberta-base```                    |
| DistilBERT   | ```distilbert-base-uncased```         |
| BART      | ```facebook/bart-base```              |
| LayoutLM  | ```microsoft/layoutlm-base-uncased``` |
