# Explainable-Latent-Structures-Using-Attention

## To Install Requirement Files

```python
pip3 install -r Requirements.txt
```
## To Download dataset

| Dataset | Download Link |
| --- | --- |
| SNLI | [snli](https://nlp.stanford.edu/projects/snli/snli_1.0.zip) |
| SST | [sst](http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip) |
| QQP | [qqp](https://www.kaggle.com/c/quora-question-pairs/data) |

# To make the datafiles
1. > python3 snli/build_vocab.py --data-paths train.jsonl:test.jsonl:valid.jsonl --out Vocab/vocab.pkl

2. > python3 dump_dataset.py --data train.jsonl --vocab Vocab/vocab.pkl --vocab-size 1000000 --max-length 100 --out train.pkl

## To Train 

```python
CUDA_LAUNCH_BLOCKING=1 python3 train.py --train-data snli/Data/train.pkl --valid-data snli/Data/val.pkl --glove snli/glove.840B.300d.txt --save-dir snli/SaveModel/dir/ --word-dim 300 --hidden-dim 300 --clf-hidden-dim 1024 --clf-num-layers 1 --batch-size 32 --max-epoch 20 --device cuda --intra-attention --batch-norm --bidirectional  --dropout 0.14
```

> **Note**: Use the pretrained Directory to access the pretrained model and vocab file for the SNLI task as described in the paper

## To Test

```python
python3 evaluate.py --model pretrained/model.pkl --data snli/Data/test.pkl --word-dim 300 --hidden-dim 300 --clf-hidden-dim 1024 --clf-num-layers 1 --leaf-rnn --batchnorm --dropout 0.1 --device cuda:0 --batch-size 32
```

> **Note** :  Use the rephrase.ipynb to see the other utils of the project. 
