# Wikipedia — NLP Pipeline: Word & URL Embeddings

Two models built on Wikipedia data: a GloVe-inspired word embedding model trained on token co-occurrence, and an LSTM-based text-to-URL retrieval model.

**Reference:** Pennington et al., *GloVe: Global Vectors for Word Representation* (EMNLP 2014)

## Setup

### System Dependencies

```bash
sudo apt install default-jre
```

### Python Environment

```bash
conda create -n esrecsys python=3.11 -y
conda activate esrecsys
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install wandb absl-py numpy nltk pyspark protobuf

cd wikipedia
pip install -r requirements.txt
```

## Data Pipeline

### 1. Download Wikipedia XML

```bash
wget https://dumps.wikimedia.org/enwiki/20220601/enwiki-20220601-pages-articles-multistream.xml.bz2
```

### 2. Convert XML to Protobuf

```bash
python xml2proto.py \
  --input_file=enwiki-20220601-pages-articles-multistream.xml.bz2 \
  --output_file=data/enwiki-latest-parsed
```

### 3. Tokenize (PySpark)

```bash
spark-submit --master=local[4] tokenize_wiki_pyspark.py \
  --input_file=data/enwiki-latest-parsed \
  --output_file=data/enwiki-latest-tokenized
```

### 4. Build Dictionaries

```bash
spark-submit --master=local[4] make_dictionary.py \
  --input_file=data/enwiki-latest-tokenized \
  --title_output=data/dictionaries/title.tstat.pb.b64.bz2 \
  --token_output=data/dictionaries/token.tstat.pb.b64.bz2 \
  --min_token_frequency=50 \
  --min_title_frequency=5 \
  --max_token_dictionary_size=500000 \
  --max_title_dictionary_size=5000000
```

### 5. Build Co-occurrence Matrix

```bash
spark-submit --master=local[4] make_cooccurrence.py \
  --input_file=data/enwiki-latest-tokenized \
  --token_dictionary=data/dictionaries/token.tstat.pb.b64.bz2 \
  --output_file=data/wikipedia.cooccur.pb.b64.bz2 \
  --context_window=10
```

## Training

### GloVe Word Embeddings

```bash
python train_cooccurence.py \
  --train_input_pattern="data/wikipedia.cooccur.pb.b64.bz2/part-?????.bz2" \
  --token_dictionary=data/dictionaries/token.tstat.pb.b64.bz2 \
  --embedding_dim=64 \
  --batch_size=2048 \
  --steps_per_epoch=10000 \
  --num_epochs=100 \
  --learning_rate=0.0005 \
  --terms="news,apple,computer,physics,neural,democracy,singapore,livermore"
```

### Text-to-URL Embedding

Requires sparse document and URL dice data from PySpark (`make_sparse_doc.py`, `make_dice.py`):

```bash
python train_txt2url.py \
  --url2url_train_input_pattern="data/wikipedia_url2url_dice/part-????[1-9].bz2" \
  --txt2url_train_input_pattern="data/wikipedia_txt2url/part-????[1-9].bz2" \
  --url2url_validation_input_pattern="data/wikipedia_url2url_dice/part-????0.bz2" \
  --txt2url_validation_input_pattern="data/wikipedia_txt2url/part-????0.bz2" \
  --token_dictionary=data/dictionaries/token.tstat.pb.b64.bz2 \
  --title_dictionary=data/dictionaries/title.tstat.pb.b64.bz2 \
  --rnn_size=64 \
  --sentence_length=32 \
  --url_embedding_dim=64 \
  --batch_size=1024 \
  --steps_per_epoch=4000 \
  --num_epochs=100 \
  --learning_rate=0.001
```

## Architectures

### GloVe
- Token embedding + bias with weighted MSE loss

### Text-to-URL
- LSTM encoder over word embeddings → URL embedding space
- Dual loss: text-to-URL margin loss + URL-to-URL Dice coefficient regression
- RMSprop optimizer

## Utilities

| Script | Purpose |
|---|---|
| `xml2proto.py` | Wikipedia XML → protobuf |
| `codex.py` | Protobuf inspection |
| `tokenize_wiki_pyspark.py` | Distributed tokenization |
| `make_dictionary.py` | Token/title frequency stats |
| `make_cooccurrence.py` | Co-occurrence matrix |
| `make_sparse_doc.py` | Sparse document builder |
| `make_dice.py` | Dice coefficient matrix |
| `count_terms.py` | TF-IDF computation |
| `dump_cooccurrence.py` / `dump_dice.py` | Debug utilities |
