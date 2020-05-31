<div align="center"><img src="https://github.com/stanfordnlp/stanza/raw/dev/images/stanza-logo.png" height="100px"/></div>

<h2 align="center">Training Tutorials for the Stanza Python NLP Library</h2>

This repo provides step-by-step tutorials for training models with [Stanza](https://github.com/stanfordnlp/stanza) - the official Python NLP library by the Stanford NLP Group. All neural modules in Stanza, including the tokenzier, the multi-word token (MWT) expander, the POS/morphological features tagger, the lemmatizer, the dependency parser, and the named entity tagger, can be trained with your own data.

This repo is meant to complement our [training documentation](https://stanfordnlp.github.io/stanza/training.html), by providing runnable scripts coupled with toy data that makes it much easier for users to get started with model training. To train models with your own data, you should be able to simply replace the provided toy data with your own data in the same format, and start using them with Stanza right after training.

## Environment Setup

We only support `python3`. You can set up your training environments by simply running each command below:

```sh
git clone https://github.com/yuhui-zh15/stanza-train.git
cd stanza-train
pip install -r requirements.txt
git clone https://github.com/stanfordnlp/stanza.git
cp config/config.sh stanza/scripts/config.sh
cp config/xpos_vocab_factory.py stanza/stanza/models/pos/xpos_vocab_factory.py
cd stanza
```

The [`config.sh`](config/config.sh) is used to set environment variables (e.g., data path, word vector path, etc.) for the training and testing of stanza modules.

The [`xpos_vocab_factory.py`](config/xpos_vocab_factory.py) is used to build XPOS vocabulary for our provided `UD_English-TEST` toy data. Compared with the file in downloaded Stanza repo, we only add its shorthand name (`en_test`) to the file. You can safely ignore it for now. If you want to use another dataset other than `UD_English-TEST` after running this tutorial, you can add the shorthand in the same pattern. In case you're curious, [here's how we built this file]( https://github.com/stanfordnlp/stanza/blob/master/stanza/models/pos/build_xpos_vocab_factory.py).


## Training and Evaluating Processors

Here we provide instructions for training each processor currently supported by Stanza, using the toy data in this repo as example datasets. Model performance will be printed during training. As our provided toy data only contain several sentences for demonstration purpose, you should be able to get 100% accuracy at the end of training.

#### `tokenize`

The [`tokenize`](https://stanfordnlp.github.io/stanza/tokenize.html) processor segments the text into tokens and sentences. All downstream processors which generate annotations at the token or sentence level depends on the output from this processor.

Training the `tokenize` processor currently requires the [Universal Dependencies](https://universaldependencies.org/) treebank data in both plain text and the `conllu` format, as you can find in our provided toy examples [here](data/udbase/UD_English-TEST). To train the `tokenize` processor with this toy data, run the following command:

```sh
bash scripts/run_tokenize.sh UD_English-TEST --step 500
```

Note that since this toy data is very small in scale, we are restricting the training with a very small `step` parameter. To train on your own data, you can either set a larger `step` parameter, or use the default parameter value.

#### `mwt`

The Universal Dependencies grammar defines syntatic relations between [syntactic words](https://universaldependencies.org/u/overview/tokenization.html), which, for many languages (e.g., French), are different from raw tokens as segmented from the text. For these languages, the [`mwt`](https://stanfordnlp.github.io/stanza/mwt.html) processor expands the multi-word tokens (MWT) recognized by the [`tokenize`](https://stanfordnlp.github.io/stanza/tokenize.html) processor into multiple syntactic words, paving the ways for downstream annotations.

> Note: The mwt processor is not needed and cannot be trained for languages that do not have [multi-word tokens (MWT)](https://universaldependencies.org/u/overview/tokenization.html), such as English or Chinese.

Like the `tokenize` processor, training the `mwt` processor requires UD data, in the format like our provided toy examples [here](data/udbase/UD_English-TEST). You can run the following command to train the `mwt` processor:

> Note: Our provided toy data is in English, which do not contain MWT, you should replace provided data with data in languages that contain MWT (e.g., German, French, etc.) to train `mwt` processor.

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
bash scripts/run_pos.sh UD_English-TEST --max_steps 500
```

#### Depparse

The [`Depparse`](https://stanfordnlp.github.io/stanza/depparse.html) processor provides an accurate syntactic dependency parser.

Training `Depparse` processor requires UD data and pretrained word vectors, where you can find our provided toy examples [here](data/udbase/UD_English-TEST) and [here](data/wordvec/word2vec/English), respectively. You can run the following command to train the `Depparse` processor:

```sh
bash scripts/run_depparse.sh UD_English-TEST gold --max_steps 500
```

#### NER

The [`NER`](https://stanfordnlp.github.io/stanza/ner.html) processor recognizes named entities for all token spans in the corpus.

Training `NER` processor requires BIO data and pretrained word vectors, where you can find our provided toy examples [here](data/nerbase/English-TEST) and [here](data/wordvec/word2vec/English), respectively. You can run the following command to train the `NER` processor:

```sh
bash scripts/run_ner.sh English-TEST --max_steps 500 --word_emb_dim 5
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
bash scripts/run_ner.sh English-TEST --max_steps 500 --word_emb_dim 5 --charlm --charlm_shorthand en_test --char_hidden_dim 1024
```

## Load Trained Processors

Loading the trained processor only requires the path for the trained model. Here we provide an example to load trained `Tokenize` processor:

```python
>>> import stanza
>>> nlp = stanza.Pipeline(lang='en', processors='tokenize', tokenize_model_path='saved_models/tokenize/en_test_tokenizer.pt')
```

## Contribute to the Model Zoo

After training your processors, we welcome you to release your models and contribute your models to our model zoo! You can file an issue [here](https://github.com/stanfordnlp/stanza/issues). Please clearly state your dataset, model performance and contact information, and please briefly introduce why you think your model would benefit the whole community! We will integrate your models into our official repository after verification.


