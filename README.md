# mwp-seq2seq

Apply the seq2seq model on the MAWPS dataset.

## Comparison of different models



<div align="center">

|filename|model | Encoder|Decoder|Equ. Acc|
|:-:|:-:|:-:|:-:|:-:|
|`model/vanilla.py`|Vanilla|GRU|LSTM|0.463|
|`model/bilstm.py`|Bi-LSTM|Bi-LSTM|LSTM+Attn| 0.640  |
|`model/tfm.py`|Transformer|Transformer|Transformer|0.720  |
|`model/bertgen.py`|BERTGen|BERT|Transformer| Coming Soon...|

</div>




## Usage

Take `vanilla.py` as an example, simply run the following commands in the terminal

```bash
git clone https://github.com/sonvier/mwp-seq2seq.git && cd mwp-seq2seq
mkdir params output && nohup python -um model.vanilla > ./output/train.log 2>&1 &
```

## Structure

```
.
├── README.md
├── arch
│   ├── mha.py
│   └── transformer.py
├── data
│   ├── test.json
│   └── train.json
├── data_preprocess.py
├── model
│   ├── bilstm.py
│   ├── tfm.py
│   └── vanilla.py
└── utils.py

3 directories, 10 files
```