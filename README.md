# Atlas-free Brain Network Transformer (atlas-free BNT)

Python implementation for the paper **[Atlas-free Brain Network Transformer](https://arxiv.org/pdf/2510.03306)**, supporting:

- **Sex classification** (`--task cls`)
- **Brain-age prediction** (`--task reg`)

* If you use this code and find it helpful, please cite the above paper. Thanks :smile:
```
@article{huang2025atlas_free,
  title={Atlas-free Brain Network Transformer},
  author={Huang, Shuai and Kan, Xuan and Lah, James J and Qiu, Deqiang},
  journal={arXiv preprint arXiv:2510.03306},
  year={2025}
}
```

## What this repo does

- Reads **per-subject cluster features** `G` (MAT file, shape `[a, b]`, where `a` varies by subject)
- Reads **per-subject 3D cluster index volumes** `C` (MAT file, shape `[X,Y,Z]`)
  - `C` values are in `[0, a]`
  - `0` = background
  - `1..a` = cluster IDs
- Builds overlapping 3D blocks and constructs a **block index matrix** `D` of shape `[num_blocks, block_size^3]`
- Mean-pools cluster embeddings within each block (background included) to obtain block/node features
- Runs a transformer encoder + DEC/OCR readout for downstream tasks (classification or regression)

## Install

```bash
pip install -r requirements.txt
```

## Data preparation

You need three aligned inputs:

1. `features_list.txt`: one MAT path per subject (feature matrices `G`)
2. `clusters_list.txt`: one MAT path per subject (3D cluster volumes `C`)
3. `labels.npy`: length-N array of classification labels (0/1), or legenth-N array of regression values, in the same order as the list files
4. `mask.npy`: 3D brain mask (same spatial shape as `C`) used to generate block coordinates

## Train with cross-validation

Quick start scripts (edit paths inside):

```bash
bash run_one_fold_abcd.sh
bash run_one_fold_age_abcd.sh
```
Details about the configuration parameters can be found in `train.py`

Sex classification example:

```bash
python train.py \
  --task cls \
  --features-list /path/to/features_list.txt \
  --clusters-list /path/to/clusters_list.txt \
  --labels-npy /path/to/labels.npy \
  --mask-npy /path/to/mni_mask.npy \
  --init-dim feature_dimensionality \
  --n-splits 10 --fold 0 --val-ratio 0.1 \
  --block-size 3 --stride 2 --block-thd 4 \
  --loss ce --device cuda:0 \
  --epochs 25 --batch-size 16 \
  --lr 1e-6 \
  --lr-step-size 20 --lr-gamma 0.5 \
  --hidden-dim 500 --num-heads 4 --num-layers 1 --dec-encoder-hidden-size 32 --dec-clusters 50 \
  --pooling mean \
  --outdir runs/sex_classification/fold0 \
  --save-preds
```

Brain-age prediction example:

```bash
python train.py \
  --task reg \
  --features-list /path/to/features_list.txt \
  --clusters-list /path/to/clusters_list.txt  \
  --labels-npy /path/to/age.npy \
  --mask-npy /path/to/mni_mask.npy \
  --init-dim feature_dimensionality \
  --n-splits 10 --fold 0 --val-ratio 0.1 \
  --block-size 3 --stride 2 --block-thd 4 \
  --loss mse --device cuda:0 \
  --epochs 50 --batch-size 16 \
  --lr-step-size 20 --lr-gamma 0.5 \
  --lr 1e-6 \
  --hidden-dim 500 --num-heads 4 --num-layers 1 --dec-encoder-hidden-size 32 --dec-clusters 50 \
  --pooling mean \
  --outdir runs/brain_age_prediction/fold0 \
  --save-preds
```

Outputs are written to `--outdir`:
- `run.log`
- `history.json`
- `summary.json`
- `best.pt`
- optional `val_logits_best.npy`, `test_logits_best.npy`, etc.


## License

MIT License (see `LICENSE`).
