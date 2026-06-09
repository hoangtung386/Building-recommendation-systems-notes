# Pinterest — Shop The Look: Visual Complementary Product Recommendation

A two-tower CNN model that recommends matching product images given a scene image (e.g., suggest furniture for a living room photo), trained with triplet loss.

**Reference:** Kang et al., *Complete the Look: Scene-based Complementary Product Recommendation* (CVPR 2019)

## Setup

```bash
conda create -n esrecsys python=3.11 -y
conda activate esrecsys
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install wandb absl-py numpy Pillow

cd pinterest
pip install -r requirements.txt
```

## Data

This project uses the [Shop The Look Dataset](https://github.com/kang205/STL-Dataset).

A copy is included in `STL-Dataset/`. To fetch images from Pinterest:

```bash
python fetch_images.py \
  --input_file=STL-Dataset/fashion.json \
  --output_dir=images/ \
  --max_lines=100000 \
  --sleep_time=5
```

Images can also be downloaded from Weights & Biases:

```bash
wandb artifact get building-recsys/recsys-pinterest/shop_the_look:latest
```

## Training

```bash
python train_shop_the_look.py \
  --input_file=STL-Dataset/fashion.json \
  --image_dir=./artifacts/shop_the_look:v1 \
  --max_steps=30000 \
  --learning_rate=0.0001 \
  --regularization=0.2 \
  --output_size=64 \
  --checkpoint_every_steps=10000 \
  --model_name=pinterest_stl_model.pt
```

Checkpoints and logs are saved via Weights & Biases.

## Inference

Generate embeddings for all scenes and products:

```bash
python make_embeddings.py \
  --input_file=STL-Dataset/fashion.json \
  --image_dir=./artifacts/shop_the_look:v1 \
  --model_name=pinterest_stl_model.pt \
  --output_size=64
```

Generate top-K recommendations:

```bash
python make_recommendations.py \
  --product_embed=product_embed.json \
  --scene_embed=scene_embed.json \
  --top_k=10
```

## Baseline

```bash
python random_item_recommender.py \
  --input_file=STL-Dataset/fashion-cat.json \
  --output_html=output.html
```

## Architecture

- **CNN**: 4 residual blocks `[16, 32, 64, 128]` with stride-2 convolutions + avg pooling
- **Two-tower**: Separate CNNs for scene and product branches
- **Loss**: Triplet loss with margin and L2 regularization on embeddings
- **Optimizer**: Adam
