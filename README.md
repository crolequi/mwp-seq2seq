# mwp-seq2seq

Apply the seq2seq model on the MAWPS dataset.

## Comparison of different models



<div align="center">

|filename|model | Encoder|Decoder|Equ. Acc|
|:-:|:-:|:-:|:-:|:-:|
|`model/vanilla.py`|Vanilla|GRU|LSTM|34.3|
|`model/bilstm.py`|Bi-LSTM|Bi-LSTM|LSTM+Attn|  |
|`model/tfm.py`|Transformer|Transformer|Transformer|  |

</div>




## Usage

Take `vanilla.py` as an example, simply run the following commands in the terminal

```bash
git clone https://github.com/sonvier/mwp-seq2seq.git && cd mwp-seq2seq
mkdir params output && nohup python -um model.vanilla > ./output/train.log 2>&1 &
```