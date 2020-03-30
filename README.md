# Stanza Training Tutorial

> Warning: only support Linux.

## Set up Environment

```sh
git clone git@github.com:yuhui-zh15/stanza-train.git
cd stanza-train
git clone git@github.com:stanfordnlp/stanza.git
cp config/config.sh stanza/scripts/config.sh
cp config/xpos_vocab_factory.py stanza/models/pos/xpos_vocab_factory.py
cd stanza
```

## Train and Evaluate Processors

#### Tokenize

```sh
bash scripts/run_tokenize.sh UD_English-TEST --step 1000
```

#### MWT

```sh
bash scripts/run_mwt.sh UD_English-TEST --num_epoch 2
```

#### POS

```sh
bash scripts/run_pos.sh UD_English-TEST --max_steps 1000
```

#### Lemma

```sh
bash scripts/run_lemma.sh UD_English-TEST --num_epoch 2
```

#### Depparse

```sh
bash scripts/run_depparse.sh UD_English-TEST gold --max_steps 1000
```

#### NER

```sh
bash scripts/run_ner.sh English-TEST --max_steps 1000 --word_emb_dim 5
```

## Load Trained Processors

```python
>>> import stanza
>>> nlp = stanza.Pipeline(lang='en', processors='tokenize,pos', tokenize_model_path='...', pos_model_path='...')
```



