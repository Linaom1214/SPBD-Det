# SPD: Decoupling Polarity with Self-Prompts for Infrared Small Target Detection

Official implementation of **SPD**, a polarity-aware infrared small-target segmentation framework for bright/dark target detection under polarity reversal and low signal-to-clutter conditions.

Project page and repository: https://github.com/Linaom1214/SPBD-Det

SPD takes a single infrared image frame as input and predicts a binary target mask. The model combines a re-parameterized infrared feature encoder, an adaptive edge denoising/enhancement module, and a self-prompt decoder with bright/dark learnable tokens to decouple target polarity during mask decoding.

## Highlights

- **Polarity-aware decoding**: bright and dark self-prompt tokens explicitly separate opposite target appearances in infrared scenes.
- **Edge-aware denoising**: a learnable Sobel/PDE-inspired module enhances weak boundaries while suppressing clutter responses.
- **Efficient deployment**: the re-parameterized encoder supports deploy-time branch fusion through `--deploy`.
- **Reproducible release**: training, evaluation, multi-seed, complexity, and inference scripts are included.

## News

- Public release repository prepared for reviewer-requested reproducibility checks.
- Supports strict reproduction of paper metrics on IRReversal/ch3 and IRTiny-BD-10K.
- Includes scripts for multi-seed reporting, complexity benchmarking, and failure-case visualization.

## Method overview

SPD is designed for binary segmentation of dim and small infrared targets whose polarity may change from bright to dark across scenes.

| Component | Configuration | Role |
| --- | --- | --- |
| Input | `1 x 3 x 512 x 512` by default | RGB-converted infrared frame with ImageNet normalization |
| Rep-EFEM encoder | multi-stage re-parameterized encoder | extracts multi-scale infrared features and supports deploy fusion |
| AEDM/EBS | learnable Sobel edge term + noise term | preserves target boundaries and suppresses edge noise |
| Self-Prompt Decoder | 3 stages, depths `[1, 2, 4]` | decouples bright/dark target features with learnable tokens |
| Output | `2 x H x W` logits | background/foreground segmentation mask |

The training objective is:

```text
L = L_CE + L_Dice + lambda_token * L_token
```

where `L_token` separates bright and dark prompt token embeddings with a cosine-distance constraint. The default `lambda_token` is `1.0`.

## Main results

Strict paper-metric reproduction uses foreground IoU/pixel accuracy and target-level probability of detection / false alarm definitions compatible with the original evaluation protocol.

| Dataset | Split size | pixAcc | mIoU | PD | FA_raw |
| --- | ---: | ---: | ---: | ---: | ---: |
| IRReversal/ch3 | 17,915 | 0.911425 | 0.851500 | 0.964890 | 4.281660e-06 |
| IRTiny-BD-10K | 2,000 | 0.939007 | 0.889078 | 0.957138 | 5.760193e-07 |

For false-alarm reporting, this repository writes `FA_raw`, `FA_x1e6`, and `FA_x1e7` to avoid ambiguity between logs and paper tables.

### Multi-seed stability on IRTiny-BD-10K

| Seed | pixAcc | mIoU | PD | FA_x1e6 |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 0.957947 | 0.915592 | 0.986539 | 0.595093 |
| 3407 | 0.932740 | 0.891469 | 0.956902 | 0.467300 |
| 2026 | 0.929859 | 0.889860 | 0.958200 | 0.726700 |
| Mean ± std | 0.940182 ± 0.012617 | 0.898974 ± 0.011769 | 0.967214 ± 0.013675 | 0.596364 ± 0.105903 |

### Complexity

Measured with input size `1 x 3 x 512 x 512`.

| Model | Params (M) | FLOPs (G) | Latency (ms) | FPS | Memory (MB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full SPD | 9.35 | 12.60 | 48.88 | 20.46 | 343.58 |
| Full SPD, deploy-fused | 9.30 | 12.36 | 42.72 | 23.41 | 343.46 |

## Repository structure

```text
configs/                  Experiment configurations
scripts/                  Multi-seed and complexity launch scripts
spdnet/                   SPD model, data pipeline, losses, metrics, optimizer
  data.py                 Binary segmentation dataset and dataloader
  losses.py               Cross-entropy, Dice, and token contrastive losses
  metrics.py              Paper-compatible and thresholded metrics
  models/spd/             Rep-EFEM encoder and Self-Prompt Decoder
tools/                    Dataset preparation, checking, summaries, visualization
train.py                  Training entry point, including DDP support
test.py                   Checkpoint evaluation and optional prediction export
inference.py              Image-folder inference
```

## Installation

A Python 3.10 environment is recommended.

```bash
conda create -n spd python=3.10 -y
conda activate spd
pip install -r requirements.txt
```

On the internal reproducibility server, commands can also be launched with:

```bash
conda run -n py310 python <command>
conda run -n py310 torchrun <torchrun-args>
```

## Dataset preparation

Datasets are expected in folder-based binary segmentation format:

```text
data/<dataset>/
  train/images/*
  train/masks/*
  val/images/*
  val/masks/*
  test/images/*
  test/masks/*
```

Dataset access:

- **IRReversal/ch3**: the dataset constructed in this work is available by contacting `linaom1214@163.com`.
- **IRTiny-BD-10K**: public benchmark dataset, available at https://github.com/kourenke/IRTiny-BD-10K-Datasets.
- **SIRST-Aug**: public benchmark dataset released with AGPCNet, available at https://github.com/Tianfang-Zhang/AGPCNet.


Requirements:

- Image and mask files should share the same file name or stem.
- Masks are binarized as foreground where pixel value is greater than `127`.
- Images are converted to RGB, resized to `512 x 512` by default, and normalized with ImageNet statistics.
- Train/val/test folders should be disjoint to avoid leakage.

Check a dataset before training:

```bash
python tools/check_dataset.py --config configs/spd_mydata.yaml
python tools/check_dataset.py --config configs/spd_irtiny_bd_10k.yaml
```

Create splits from a raw folder:

```bash
python tools/make_splits.py \
  --src-root /path/raw \
  --dst-root data/mydata \
  --train 0.7 \
  --val 0.1 \
  --test 0.2 \
  --seed 42
```

Prepare data from an existing train/test layout:

```bash
python tools/prepare_from_train_test.py \
  --src-root /path/train_test_dataset \
  --dst-root data/mydata \
  --val-ratio 0.1 \
  --seed 42
```

Use `--copy` with the split tools only if symlinks are not desired.

## Training

Train with the default configs:

```bash
python train.py --config configs/spd_mydata.yaml
python train.py --config configs/spd_irtiny_bd_10k.yaml
```

Override dataset paths or hyperparameters from the command line:

```bash
python train.py \
  --config configs/spd_mydata.yaml \
  --cfg-options data.root=/path/to/mydata train.epochs=100 seed=42
```

Resume or initialize from checkpoints:

```bash
python train.py --config configs/spd_mydata.yaml --resume work_dirs/spd_mydata/last.pth
python train.py --config configs/spd_mydata.yaml --load-from /path/to/checkpoint.pth
```

Multi-GPU training uses PyTorch DDP:

```bash
torchrun --nproc_per_node=2 train.py --config configs/spd_irtiny_bd_10k.yaml
```

Training writes configs, logs, history, curves, TensorBoard logs, and `best.pth` / `last.pth` checkpoints under `work_dirs/<experiment_name>` by default.

## Evaluation

Evaluate a checkpoint:

```bash
python test.py \
  --config configs/spd_mydata.yaml \
  --checkpoint work_dirs/spd_mydata/best.pth

python test.py \
  --config configs/spd_irtiny_bd_10k.yaml \
  --checkpoint work_dirs/spd_irtiny_bd_10k/best.pth
```

Save binary predictions:

```bash
python test.py \
  --config configs/spd_mydata.yaml \
  --checkpoint work_dirs/spd_mydata/best.pth \
  --save-pred
```

Evaluate another split or deploy-fused model:

```bash
python test.py --config configs/spd_mydata.yaml --checkpoint work_dirs/spd_mydata/best.pth --split val
python test.py --config configs/spd_mydata.yaml --checkpoint work_dirs/spd_mydata/best.pth --deploy
```

The evaluation output includes:

- `mIoU`, `PixelAcc`: paper-compatible foreground metrics.
- `argmax_*`: two-class argmax mean-IoU metrics.
- `threshold_*`: thresholded binary segmentation metrics.
- `paper_*`: strict paper metrics including `PD` and `FA`.

## Inference

Run prediction on an image folder:

```bash
python inference.py \
  --config configs/spd_mydata.yaml \
  --checkpoint work_dirs/spd_mydata/best.pth \
  --img-dir demo/images \
  --out-dir demo/predictions
```

Predicted masks are resized back to the original image size and saved with the input file names.

## Reproducing paper metrics

### IRReversal/ch3 strict split

The historical IRReversal/ch3 paper split is the 17,915-image `mulframe` test split reconstructed from saved prediction names. Do not replace it with smaller local folders when reproducing the paper table.

```bash
python tools/make_mulframe_split_from_predictions.py \
  --mulframe-root /path/to/mulframe \
  --pred-dir /path/to/original_predictions \
  --out-dir work_dirs/ch3_mulframe_split

python tools/check_dataset.py --config configs/spd_ch3_mulframe_abc.yaml

python tools/eval_saved_predictions.py \
  --pred-dir /path/to/original_predictions \
  --mask-root /path/to/mulframe/masks \
  --split-file work_dirs/ch3_mulframe_split/test_mydataset.txt \
  --out work_dirs/ch3_mulframe_split/saved_prediction_metrics.json
```

Expected strict metrics:

```text
pixAcc = 0.9114254293602883
mIoU   = 0.8514998661077805
PD     = 0.964889757186715
FA_raw = 4.281659649298772e-06
```

### IRTiny-BD-10K strict checkpoint evaluation

Expected strict metrics:

```text
pixAcc = 0.9390067207820546
mIoU   = 0.8890780470067106
PD     = 0.957137796670209
FA_raw = 5.76019287109375e-07
```

## Multi-seed experiments

```bash
bash scripts/run_multiseed.sh configs/spd_irtiny_bd_10k.yaml /path/to/IRTiny-BD-10K
python tools/summarize_multiseed.py work_dirs/*/eval_test/metrics.json
```

## Complexity benchmarking

Benchmark the full model:

```bash
python tools/model_complexity.py \
  --config configs/spd_irtiny_bd_10k.yaml \
  --warmup 20 \
  --iters 100 \
  --out work_dirs/complexity/irtiny_full.json
```

Benchmark the deploy-fused model:

```bash
python tools/model_complexity.py \
  --config configs/spd_irtiny_bd_10k.yaml \
  --deploy \
  --warmup 20 \
  --iters 100 \
  --out work_dirs/complexity/irtiny_full_deploy.json
```

Run all configured variants:

```bash
bash scripts/run_complexity.sh configs/spd_irtiny_bd_10k.yaml work_dirs/complexity
python tools/summarize_complexity.py work_dirs/complexity/*.json
```

## Reproducibility defaults

| Item | Default |
| --- | --- |
| Input size | `512 x 512` |
| Image conversion | RGB-converted infrared image |
| Normalization | mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]` |
| Mask foreground | pixel value `> 127` |
| Optimizer | AdamW |
| Learning rate | `1e-3` |
| Weight decay | `1e-3` |
| Batch size | `8` train / `8` eval by default |
| Epochs | `100` |
| Scheduler | polynomial decay, power `0.9` |
| Warmup | `5` epochs |
| Minimum LR | `1e-4` |
| Seed | `42` |
| AMP | disabled by default |
| Inference threshold | foreground probability `0.5` |
| Post-processing | none |

## Checkpoints and datasets

For a complete reproducibility package, release users should be able to obtain:

- final train/val/test split files or folder layouts;
- pretrained `best.pth` checkpoints for the reported tables;
- logs or metric JSON files corresponding to the paper results;
- dataset construction and annotation notes for IRReversal/ch3.

## Citation

If this repository is useful for your research, please cite the paper:

```bibtex
@article{spd2026,
  title   = {Decoupling Polarity with Self-Prompts: A New Framework for Bright-Dark Infrared Small Target Detection},
  author  = {Lin, Jian and Li, Shaoyi and Yang, Xi and Niu, Saisai and Yue, Xiaokui},
  journal = {Pattern Recognition},
  year    = {2026},
  note    = {Manuscript under review}
}
```

## License

This project is released under the Apache License 2.0. See [LICENSE](LICENSE) for details.

## Acknowledgements

This implementation uses PyTorch and common open-source scientific Python packages. The decoder design follows the general token-to-image / image-to-token attention style used in prompt-based segmentation decoders, adapted here for polarity-aware infrared target segmentation.
