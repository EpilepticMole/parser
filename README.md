# :rocket: SuPar VAE

A Python package designed for structured prediction trying to implement a Latent Parser using a VAE. I based my work on the Constituency Parser of Yu Zhang.

* Constituency Parser
  * CRF ([Zhang et al., 2020b](https://www.ijcai.org/Proceedings/2020/560/))

* Tree
  * ConstituencyCRF ([Stern et al. 2017](https://aclanthology.org/P17-1076); [Zhang et al., 2020b](https://www.ijcai.org/Proceedings/2020/560/))

## Installation

You can install `SuPar` from source directly:
```sh
$ pip install -U git+https://github.com/EpilepticMole/parser
```

The following requirements should be satisfied:
* `python`: >= 3.8
* [`pytorch`](https://github.com/pytorch/pytorch): >= 1.8
* [`transformers`](https://github.com/huggingface/transformers): >= 4.0

## Usage

You can download the pretrained model and parse sentences with just a few lines of code:
```py
>>> from supar import Parser
# if the gpu device is available
# >>> torch.cuda.set_device('cuda:0')  
>>> parser = Parser.load('con-crf-en')
>>> dataset = parser.predict('I saw Sarah with a telescope.', lang='en', prob=True, verbose=False)
```
By default, the parser use [`stanza`](https://github.com/stanfordnlp/stanza) internally to tokenize plain texts for parsing.
You only need to specify the language code `lang` for tokenization.

The call to `parser.predict` will return an instance of `supar.utils.Dataset` containing the predicted results.
You can either access each sentence held in `dataset` or an individual field of all results.
Probabilities can be returned along with the results if `prob=True`.
```py
>>> dataset[0]
1       I       _       _       _       _       2       nsubj   _       _
2       saw     _       _       _       _       0       root    _       _
3       Sarah   _       _       _       _       2       dobj    _       _
4       with    _       _       _       _       2       prep    _       _
5       a       _       _       _       _       6       det     _       _
6       telescope       _       _       _       _       4       pobj    _       _
7       .       _       _       _       _       2       punct   _       _

>>> print(f"arcs:  {dataset.arcs[0]}\n"
          f"rels:  {dataset.rels[0]}\n"
          f"probs: {dataset.probs[0].gather(1,torch.tensor(dataset.arcs[0]).unsqueeze(1)).squeeze(-1)}")
arcs:  [2, 0, 2, 2, 6, 4, 2]
rels:  ['nsubj', 'root', 'dobj', 'prep', 'det', 'pobj', 'punct']
probs: tensor([1.0000, 0.9999, 0.9966, 0.8944, 1.0000, 1.0000, 0.9999])
```

`SuPar` also supports parsing from tokenized sentences or from file.
For BiLSTM-based semantic dependency parsing models, lemmas and POS tags are needed.

```py
>>> import os
>>> import tempfile
# if the gpu device is available
# >>> torch.cuda.set_device('cuda:0')

>>> con = Parser.load('con-crf-en')
>>> con.predict(['I', 'saw', 'Sarah', 'with', 'a', 'telescope', '.'], verbose=False)[0].pretty_print()
              TOP                       
               |                         
               S                        
  _____________|______________________   
 |             VP                     | 
 |    _________|____                  |  
 |   |    |         PP                | 
 |   |    |     ____|___              |  
 NP  |    NP   |        NP            | 
 |   |    |    |     ___|______       |  
 _   _    _    _    _          _      _ 
 |   |    |    |    |          |      |  
 I  saw Sarah with  a      telescope  . 
```

### Training

To train a model from scratch, it is preferred to use the command-line option, which is more flexible and customizable.
Below is an example of training Biaffine Dependency Parser:
```sh
$ python -m supar.cmds.const.crf train -b -d 0 -c con-crf-en -p model -f char
```

```sh
$ con-crf train -b -d 0 -c con-crf-en -p model -f char
```

To accommodate large models, distributed training is also supported:
```sh
$ python -m supar.cmds.const.crf train -b -c con-crf-en -d 0,1,2,3 -p model -f char
```

### Evaluation

The evaluation process resembles prediction:
```py
# if the gpu device is available
# >>> torch.cuda.set_device('cuda:0')  
>>> Parser.load('con-crf-en').evaluate('ptb/test.conllx', verbose=False)
loss: 0.2393 - UCM: 60.51% LCM: 50.37% UAS: 96.01% LAS: 94.41%
```

See [examples](examples) for more instructions on training and evaluation.

## Performance

`SuPar` provides pretrained models for English, Chinese and 17 other languages.
The tables below list the performance and parsing speed of pretrained models for different tasks.
All results are tested on the machine with Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz and Nvidia GeForce GTX 1080 Ti GPU.

### Constituency Parsing

We use PTB and CTB7 datasets to train English and Chinese constituency parsing models.
Below are the results.

| Name                 |   P   |   R   | F<sub>1 | Sents/s |
| -------------------- | :---: | :---: | :-----: | ------: |
| `con-crf-en`         | 94.16 | 93.98 |  94.07  |  841.88 |
| `con-crf-roberta-en` | 96.42 | 96.13 |  96.28  |  233.34 |
| `con-crf-zh`         | 88.82 | 88.42 |  88.62  |  590.05 |
| `con-crf-electra-zh` | 92.18 | 91.66 |  91.92  |  140.45 |

The multilingual model `con-crf-xlmr` is trained on SPMRL dataset by finetuning [`xlm-roberta-large`](https://huggingface.co/xlm-roberta-large).
It follows instructions of [Benepar](https://github.com/nikitakit/self-attentive-parser) to preprocess the data.
For simplicity, it then directly merge train/dev/test treebanks of all languages in SPMRL into big ones to train the model.
The results of each treebank are as follows.

| Language |   P   |   R   | F<sub>1 | Sents/s |
| -------- | :---: | :---: | :-----: | ------: |
| `eu`     | 93.40 | 94.19 |  93.79  |  266.96 |
| `fr`     | 88.77 | 88.84 |  88.81  |  149.34 |
| `de`     | 93.68 | 92.18 |  92.92  |  200.31 |
| `he`     | 94.65 | 95.20 |  94.93  |  172.50 |
| `hu`     | 96.70 | 96.81 |  96.76  |  186.58 |
| `ko`     | 91.75 | 92.46 |  92.11  |  234.86 |
| `pl`     | 97.33 | 97.27 |  97.30  |  310.86 |
| `sv`     | 92.51 | 92.50 |  92.50  |  235.49 |
