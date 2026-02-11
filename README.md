# English-to-French Machine Translation on the Europarl Corpus

A comparative study of three machine translation approaches for English-to-French, developed as part of an NLP course project at Universit&eacute; Paris-Saclay.

## Overview

We implement and compare three translation systems of increasing complexity:

1. **Word-for-Word Baseline** -- Builds a positional bilingual dictionary from aligned training pairs and translates word by word.
2. **Cross-Lingual Embedding Retrieval** -- Uses multilingual sentence embeddings to retrieve the most semantically similar French sentence from a candidate pool.
3. **Fine-Tuned Seq2Seq Transformer** -- Fine-tunes a pretrained MarianMT model on the Europarl training data with hyperparameters selected via grid search.

All models are evaluated on a held-out test set using BLEU score. The fine-tuned transformer significantly outperforms both baselines (BLEU 0.32 vs. 0.05 and 0.04).

## Dataset

We use the [Europarl v8](https://opus.nlpl.eu/datasets/Europarl?pair=en&fr) parallel corpus from the OPUS collection, which contains sentence-aligned proceedings of the European Parliament. The English-French pair provides large-scale, high-quality parallel text well suited for training and evaluating translation systems.

To set up the data, download the Europarl v8 bilingual TMX file (`en-fr.tmx`) from the link above and place the extracted files in the `data/` directory.

## Project Structure

```
├── machine_translation.ipynb   # Main notebook: training, evaluation, and analysis
├── data_analysis.ipynb         # Exploratory data analysis
├── data/                       # Europarl parallel corpus files
├── seq2seq_model_opus-mt-en-fr/# Fine-tuned model checkpoints
├── logs/                       # Training logs
└── report/                     # ACL-style project report (LaTeX)
```

## Authors

Maxime Hayakawa, Juan Kenichi Sutan, Amine Outmani
