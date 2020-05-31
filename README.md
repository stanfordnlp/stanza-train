<div align="center"><img src="https://github.com/stanfordnlp/stanza/raw/dev/images/stanza-logo.png" height="100px"/></div>

<h2 align="center">Stanza Training Tutorial</h2>

This repo provides a tutorial for training [Stanza](https://github.com/stanfordnlp/stanza) --- The Stanford NLP Group's official Python NLP library. All neural modules in Stanza, including the tokenzier, the multi-word token (MWT) expander, the POS/morphological features tagger, the lemmatizer, the dependency parser, and the named entity tagger, can be trained with your own data. 

Complemented with our [training documentation](https://stanfordnlp.github.io/stanza/training.html), this repo provides runnable codes and toy data to make it much easier for users to get started with model training. With this repo, you can simply replace provided toy data with your own data to train your own modules and load them with Stanza! 

## Set up Environment

You should first install `python 3` with the following dependencies: `numpy`, `protobuf`, `requests`, `tqdm`, `torch>=1.3.0`. Then you can set up your training environments by simply running each command below. 

```sh
git clone https://github.com/yuhui-zh15/stanza-train.git
cd stanza-train
git clone https://github.com/stanfordnlp/stanza.git
cp config/config.sh stanza/scripts/config.sh
cp config/xpos_vocab_factory.py stanza/stanza/models/pos/xpos_vocab_factory.py
cd stanza
```

## Train and Evaluate Processors

Here we provide training tutorials for each processor. Model performance will be printed during training. As our provided data only contain several sentences for demonstration purpose, you should end up with 100% accuracy after training each processor.

#### Tokenize

The [`Tokenize`](https://stanfordnlp.github.io/stanza/tokenize.html) processor tokenizes the text and performs sentence segmentation, so that downstream annotation can happen at the sentence level. 

Training `Tokenize` processor requires UD data, where you can find our provided toy examples [here](data/udbase/UD_English-TEST). You can run the following command to train the `Tokenize` processor:

```sh
bash scripts/run_tokenize.sh UD_English-TEST --step 1000
```

#### MWT

The [`MWT`](https://stanfordnlp.github.io/stanza/mwt.html) Processor expands multi-word tokens (MWT) predicted by the [`Tokenize`](https://stanfordnlp.github.io/stanza/tokenize.html) Processor.

> Note: Only languages with [multi-word tokens (MWT)](https://universaldependencies.org/u/overview/tokenization.html) require MWTProcessor.

Training `MWT` processor requires UD data, where you can find our provided toy examples [here](data/udbase/UD_English-TEST). You can run the following command to train the `MWT` processor:

> Note: Our provided toy data is in English, which do not contain MWT, you should replace provided data with data in languages that contain MWT (e.g., German, French, etc.) to train `MWT` processor.

```sh
bash scripts/run_mwt.sh UD_English-TEST --num_epoch 2
```

#### Lemma

The [`Lemma`](https://stanfordnlp.github.io/stanza/lemma.html) processor generates the word lemmas for all tokens in the corpus.

Training `Lemma` processor requires UD data, where you can find our provided toy examples [here](data/udbase/UD_English-TEST). You can run the following command to train the `Lemma` processor:

```sh
bash scripts/run_lemma.sh UD_English-TEST --num_epoch 2
```

#### POS


The [`POS`](https://stanfordnlp.github.io/stanza/lemma.html) processor labels tokens with their [universal POS (UPOS) tags](https://universaldependencies.org/u/pos/), treebank-specific POS (XPOS) tags, and [universal morphological features (UFeats)](https://universaldependencies.org/u/feat/index.html).

Training `POS` processor requires UD data and pretrained word vectors, where you can find our provided toy examples [here](data/udbase/UD_English-TEST) and [here](data/wordvec/word2vec/English), respectively. You can run the following command to train the `POS` processor:

```sh
bash scripts/run_pos.sh UD_English-TEST --max_steps 1000
```

#### Depparse

The [`Depparse`](https://stanfordnlp.github.io/stanza/depparse.html) processor provides an accurate syntactic dependency parser.

Training `Depparse` processor requires UD data and pretrained word vectors, where you can find our provided toy examples [here](data/udbase/UD_English-TEST) and [here](data/wordvec/word2vec/English), respectively. You can run the following command to train the `Depparse` processor:

```sh
bash scripts/run_depparse.sh UD_English-TEST gold --max_steps 1000
```

#### NER

The [`NER`](https://stanfordnlp.github.io/stanza/ner.html) processor recognizes named entities for all token spans in the corpus.

Training `NER` processor requires BIO data and pretrained word vectors, where you can find our provided toy examples [here](data/nerbase/English-TEST) and [here](data/wordvec/word2vec/English), respectively. You can run the following command to train the `NER` processor:

```sh
bash scripts/run_ner.sh English-TEST --max_steps 1000 --word_emb_dim 5
```

#### Contextualized NER 


The performance of [`NER`](https://stanfordnlp.github.io/stanza/ner.html) processor can be significantly improved by using contextualized string representation-based sequence tagger. To enable contextualized string representation, first you need to train bidirectional character-level language models (CharLM), and then adopt the pretrained CharLM to enhance string representation.


##### CharLM


Training `CharLM` requires a large amount of raw text, where you can find our provided toy examples [here](data/processed/charlm/English/test). You can run the following command to train the forward and backward `CharLM`, respectively:

```sh
bash scripts/run_charlm.sh English-TEST forward --epochs 2 --cutoff 0 --batch_size 2
bash scripts/run_charlm.sh English-TEST backward --epochs 2 --cutoff 0 --batch_size 2
```

##### NER with CharLM

Training contextualized `NER` processor not only requires BIO data and pretrained word vectors, where you can find our provided toy examples [here](data/nerbase/English-TEST) and [here](data/wordvec/word2vec/English), respectively, it also requires pretrained CharLMs, which can be obtained by previous training step and will be saved at `saved_models/charlm`. You can run the following command to train the contextualized `NER` processor:

```sh
bash scripts/run_ner.sh English-TEST --max_steps 1000 --word_emb_dim 5 --charlm --charlm_shorthand en_test --char_hidden_dim 1024
```

## Load Trained Processors

Loading the trained processor only requires the path for the trained model. Here we provide an example to load trained `Tokenize` processor:

```python
>>> import stanza
>>> nlp = stanza.Pipeline(lang='en', processors='tokenize', tokenize_model_path='saved_models/tokenize/en_test_tokenizer.pt')
```

## Contribute to the Model Zoo

After training your processors, we welcome you to release your models and contribute your models to our model zoo! You can file up an issue [here](https://github.com/stanfordnlp/stanza/issues). Please clearly state your dataset and model performance, and briefly introduce why you think your model would benefit the whole community! We will integrate your models into our official repository after verification.


