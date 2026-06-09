# Spotify — Million Playlist Dataset: Playlist Continuation

An embedding-based retrieval model that predicts which tracks come next given a playlist context (first 5 tracks), using album and artist embeddings with contrastive learning.

**Reference:** [Spotify Million Playlist Dataset](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge)

## Setup

```bash
conda create -n esrecsys python=3.11 -y
conda activate esrecsys
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install wandb absl-py numpy

cd spotify
pip install -r requirements.txt
```

## Data

Download the Million Playlist Dataset from [AIcrowd](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge/dataset_files):

```bash
mkdir -p data/spotify_million_playlist_dataset
mkdir data/spotify_million_playlist_dataset_challenge

# Unpack into the directories above
```

## Data Preparation

Build dictionaries from playlist JSON files:

```bash
python make_dictionary.py \
  --playlists=data/spotify_million_playlist_dataset/data/mpd.slice*.json \
  --output=data/dictionaries
```

Generate training data (JSONL format):

```bash
python make_training.py \
  --playlists=data/spotify_million_playlist_dataset/data/mpd.slice.*.json \
  --dictionaries=data/dictionaries \
  --output=data/training
```

Pre-built dictionaries can be fetched from W&B:

```bash
wandb artifact get --type dictionaries recsys-spotify/dictionaries
```

## Training

```bash
python train_spotify.py \
  --train_pattern="data/training/00??[0-8].jsonl" \
  --test_pattern="data/training/00??9.jsonl" \
  --all_tracks=data/training/all_tracks.json \
  --dictionaries=data/dictionaries \
  --num_negatives=64 \
  --learning_rate=1e-3 \
  --momentum=0.98 \
  --feature_size=32 \
  --max_steps=100000
```

## Architecture

- **Track embedding**: `concat(album_embedding, artist_embedding)` via hash embedding tables
- **Score**: Max dot-product of context embeddings with candidate embeddings + album/artist bonus
- **Loss**: Mean triplet + extremal triplet + L2 regularization + self-affinity penalties
- **Optimizer**: SGD with momentum
- **Evaluation**: Recall@500 for tracks and artists
