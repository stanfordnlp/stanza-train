<div align="center"><img src="https://github.com/stanfordnlp/stanza/raw/dev/images/stanza-logo.png" height="100px"/></div>

<h2 align="center">Training Tutorials for the Stanza Python NLP Library</h2>

This repo provides step-by-step tutorials for training models with [Stanza](https://github.com/stanfordnlp/stanza) - the official Python NLP library by the Stanford NLP Group. All neural processors in Stanza, including the tokenzier, the multi-word token (MWT) expander, the POS/morphological features tagger, the lemmatizer, the dependency parser, and the named entity tagger, can be trained with your own data.

This repo is meant to complement our [training documentation](https://stanfordnlp.github.io/stanza/training.html), by providing runnable scripts coupled with toy data that makes it much easier for users to get started with model training. To train models with your own data, you should be able to simply replace the provided toy data with your own data in the same format, and start using them with Stanza right after training.

> Warning: This repo is fully tested on Linux. Due to syntax differences between macOS and Linux (e.g., the `declare -A` in the `scripts/treebank_to_shorthand.sh` is not supported by macOS), you need to rewrite some files to run on macOS.  The command lines given here will not work on Windows.

This repo is designed for and tested on stanza 1.4.0.  Earlier versions will not fully work with these commands.

### Windows

To reiterate, this is only tested on Linux.  In order to run on
Windows, there is a `source scripts/config.sh` line in the initial
setup below.  Theoretically, if you manually set those variables in
the shell, or if you add those variables to the environment using the
control panel, the rest of the scripts might work.

## Environment Setup

Run the following commands at the command line.

Stanza only supports `python3`. You can install all dependencies needed by training Stanza models with:
```bash
pip install -r requirements.txt
```

Next, set up the folders and scripts needed for training with:

```bash
git clone git@github.com:stanfordnlp/stanza-train.git
cd stanza-train

git clone git@github.com:stanfordnlp/stanza.git
cp config/config.sh stanza/scripts/config.sh
cp config/xpos_vocab_factory.py stanza/stanza/models/pos/xpos_vocab_factory.py
cd stanza
source scripts/config.sh
```

The [`config.sh`](config/config.sh) script is used to set environment variables (e.g., data path, word vector path, etc.) needed by training and testing Stanza models.

The [`xpos_vocab_factory.py`](config/xpos_vocab_factory.py) script is used to build XPOS vocabulary file for our provided `UD_English-TEST` toy treebank. Compared with the original file in the downloaded Stanza repo, we only add the shorthand name of the toy treebank (`en_test`) to the script, so that it can be recognized during training. If you want to use another dataset other than `UD_English-TEST` after running this tutorial, you can add the shorthand of your treebank in the same way. In case you're curious, [here's how we built this file]( https://github.com/stanfordnlp/stanza/blob/master/stanza/models/pos/build_xpos_vocab_factory.py).


## Training and Evaluating Processors

Here we provide instructions for training each processor currently supported by Stanza, using the toy data in this repo as example datasets. Model performance will be printed during training. As our provided toy data only contain several sentences for demonstration purpose, you should be able to get 100% accuracy at the end of training.

### `tokenize`

The [`tokenize`](https://stanfordnlp.github.io/stanza/tokenize.html) processor segments the text into tokens and sentences. All downstream processors which generate annotations at the token or sentence level depends on the output from this processor.

Training the `tokenize` processor currently requires the [Universal Dependencies](https://universaldependencies.org/) treebank data in both plain text and the `conllu` format, as you can find in our provided toy examples [here](data/udbase/UD_English-TEST). To train the `tokenize` processor with this toy data, run the following command:

```sh
python3 -m stanza.utils.datasets.prepare_tokenizer_treebank UD_English-TEST
python3 -m stanza.utils.training.run_tokenizer UD_English-TEST --step 500
```

Note that since this toy data is very small in scale, we are restricting the training with a very small `step` parameter. To train on your own data, you can either set a larger `step` parameter, or use the default parameter value.

### `mwt`

The Universal Dependencies grammar defines syntatic relations between [syntactic words](https://universaldependencies.org/u/overview/tokenization.html), which, for many languages (e.g., French), are different from raw tokens as segmented from the text. For these languages, the [`mwt`](https://stanfordnlp.github.io/stanza/mwt.html) processor expands the multi-word tokens (MWT) recognized by the [`tokenize`](https://stanfordnlp.github.io/stanza/tokenize.html) processor into multiple syntactic words, paving the ways for downstream annotations.

> Note: The mwt processor is not needed and cannot be trained for languages that do not have [multi-word tokens (MWT)](https://universaldependencies.org/u/overview/tokenization.html), such as English or Chinese.

Like the `tokenize` processor, training the `mwt` processor requires UD data, in the format like our provided toy examples [here](data/udbase/UD_English-TEST). You can run the following command to train the `mwt` processor:

```sh
python3 -m stanza.utils.datasets.prepare_mwt_treebank UD_English-TEST
python3 -m stanza.utils.training.run_mwt UD_English-TEST --num_epoch 2
```

> Note: Running the above command with the toy data will yield a message saying that zero training data can be found for MWT training. This is normal since MWT is not needed for English. The training should work when you replace the provided data with data in languages that support MWT (e.g., German, French, etc.).

### `lemma`

The [`lemma`](https://stanfordnlp.github.io/stanza/lemma.html) processor predicts lemmas for all words in an input sentence. Training the `lemma` processor requires data files in the `conllu` format. With the toy examples, you can train the `lemma` processor with the following command:

```sh
python3 -m stanza.utils.datasets.prepare_lemma_treebank UD_English-TEST
python3 -m stanza.utils.training.run_lemma UD_English-TEST --num_epoch 2
```

### `pos`

The [`pos`](https://stanfordnlp.github.io/stanza/lemma.html) processor annotates words with three types of syntactic information simultaneously: the [Universal POS (UPOS) tags](https://universaldependencies.org/u/pos/), and treebank-specific POS (XPOS) tags, and [universal morphological features (UFeats)](https://universaldependencies.org/u/feat/index.html).

Training the `pos` processor usually requires UD data in the `conllu` format and pretrained word vectors. For demo purpose, we provide an example word vector file [here](data/wordvec/word2vec/English). With the toy data and word vector file, you can train the `pos` processor with:

```sh
python3 -m stanza.utils.datasets.prepare_pos_treebank UD_English-TEST
python3 -m stanza.utils.training.run_pos UD_English-TEST --max_steps 500
```

### `depparse`

The [`depparse`](https://stanfordnlp.github.io/stanza/depparse.html) processor implements a dependency parser that predicts syntactic relations between words in a sentence. Training the `depparse` processor requires data files in the `conllu` format, and a pretrained word vector file. With the toy data and word vector file, you can train the `depparse` processor with:

```sh
python3 -m stanza.utils.datasets.prepare_depparse_treebank UD_English-TEST
python3 -m stanza.utils.training.run_depparse UD_English-TEST --max_steps 500
```

Note that the `gold` parameter here tells the scripts to use the "gold" human-annotated POS tags in the training of the parser.

### `ner`

The [`ner`](https://stanfordnlp.github.io/stanza/ner.html) processor recognizes named entities in the input text. Training the `ner` processor requires column training data in either `BIO` or `BIOES` format. See [this wikipedia page](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)) for an introduction of the formats. We provide toy examples [here](data/nerbase/English-TEST) in the BIO format. For better performance a pretrained word vector file is also recommended. With the toy data and word vector file, you can train the `ner` processor with:

```sh
python3 -m stanza.utils.training.run_ner en_sample --max_steps 500 --word_emb_dim 5
```

Note that for demo purpose we are restricting the word vector dimension to be 5 with the `word_emb_dim` parameter. You should change it to match the dimension of your own word vectors.


### Improving NER Performance with Contextualized Character Language Models

The performance of the [`ner`](https://stanfordnlp.github.io/stanza/ner.html) processor can be significantly improved by using contextualized string embeddings (i.e., a character-level language model), as was shown in [this COLING 2018 paper](https://www.aclweb.org/anthology/C18-1139/). To enable this in your NER model, you'll need to first train two character-level language models for your language (named as `charlm` module in Stanza), and then use these trained `charlm` models in your NER training.


#### `charlm`

Training `charlm` requires a large amount of raw text, such as text from news articles or wikipedia pages, in plain text files. We provide toy data for training `charlm` [here](data/processed/charlm/English/test). With the toy data, you can run the following command to train two `charlm` models, one in the forward direction of the text and another in the backward direction, respectively:

```sh
python3 -m stanza.utils.training.run_charlm en_TEST --forward   --epochs 2 --cutoff 0 --batch_size 2
python3 -m stanza.utils.training.run_charlm en_TEST --backward  --epochs 2 --cutoff 0 --batch_size 2
```

Running these commands will result in two model files in the `saved_models/charlm` directory, with the prefix `en_test`.

> Note: For details on why two models are needed and how they are used in the NER tagger, please refer to [this COLING 2018 paper](https://www.aclweb.org/anthology/C18-1139/).

#### Training contextualized `ner` models with pretrained `charlm`

Training contextualized `ner` models requires BIO-format data, pretrained word vectors, and the pretrained `charlm` models obtained in the last step. You can run the following command to train the `ner` processor:

```sh
python3 -m stanza.utils.training.run_ner en_sample --max_steps 500 --word_emb_dim 5 --charlm test
```

Note that the `charlm` here instructs the training script to look for the character language model files with the prefix of `en_test`.


## Initializing Processors with Trained Models

Initializing a processor with your own trained model only requires the path for the model file. Here we provide an example to initialize the `tokenize` processor with a model file saved at `saved_models/tokenize/en_test_tokenizer.pt`:

```python
>>> import stanza
>>> nlp = stanza.Pipeline(lang='en', processors='tokenize', tokenize_model_path='saved_models/tokenize/en_test_tokenizer.pt')
```

## Contributing Your Models to the Model Zoo

After training your own models, we welcome you to contribute your models so that it can be used by the community. To do this, you can start by creating a [GitHub issue](https://github.com/stanfordnlp/stanza/issues). Please help us understand your model by clearly describing your dataset, model performance, your contact information, and why you think your model would benefit the whole community. We will integrate your models into our official repository once we are able to verify its quality and usability.
