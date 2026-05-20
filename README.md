# Signing Conversion

Stage 1 singing voice conversion pipeline based on SoftVC-style acoustic modeling.

The project extracts acoustic features from singing voice, trains a Stage 1 acoustic model, and relies on a pretrained HiFi-GAN vocoder for waveform synthesis.

## Overview

```text
content + log-F0 + voiced + volume + speaker-id
  -> ConditioningFusion
  -> SoftVC encoder (conv + x2 upsample)
  -> SoftVC autoregressive LSTM decoder
  -> mel
  -> HiFi-GAN
```

Main acoustic model:

```text
src/svc/models/acoustic/softvc_f0.py
```

## Pipeline Architecture

```text
Raw singing audio
  -> resample to 16 kHz
  -> HuBERT soft content units
  -> RMVPE log-F0 + voiced flag
  -> RMS volume
  -> log-mel target
  -> JSONL manifests
```

Training:

```text
manifest
  -> content/f0/voiced/volume/mel features
  -> Stage 1 acoustic model
  -> masked L1 mel loss
  -> checkpoint
```

Inference:

```text
source wav
  -> content + log-F0 + voiced + volume
  -> Stage 1 acoustic model
  -> generated mel
  -> pretrained HiFi-GAN
  -> converted wav
```

## Contribution

This project does not reimplement HuBERT, RMVPE, or HiFi-GAN. It builds a focused Stage 1 training pipeline around these existing components.

The main contribution is an acoustic model for singing voice conversion that extends SoftVC-style content conditioning with explicit pitch, voicing, loudness, and speaker information:

```text
soft content units + log-F0 + voiced + volume + speaker-id -> mel
```

This conditioning is important for singing, where pitch continuity, voiced/unvoiced decisions, and dynamic energy strongly shape vocal identity.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[rmvpe,tracking]'
```

On Colab/Linux CUDA:

```bash
python -m pip install -e '.[rmvpe-cuda,tracking]'
```

Without Weights & Biases:

```bash
python -m pip install -e '.[rmvpe]'
```

## Data Layout

```text
data/raw/m4singer/
  train/<speaker>/*.wav
  dev/<speaker>/*.wav
  test/<speaker>/*.wav
```

## Prepare M4Singer

```bash
python scripts/prepare_m4singer_dataset.py --config configs/dataset/m4singer.yaml
```

## Preprocess


```bash
python -m svc.cli.cache --config configs/prepare/manifest.yaml
```

```bash
python -m svc.cli.prepare --config configs/prepare/manifest.yaml
```

Outputs:

```text
data/processed/
data/manifests/manifest_train.jsonl
data/manifests/manifest_dev.jsonl
```

## Train

```bash
python -m svc.cli.train --config configs/train/softvc_f0.yaml
```

## Convert

```bash
python -m svc.cli.convert \
  --config configs/inference/softvc_f0.yaml \
  --checkpoint checkpoints/softvc_f0_last.pt \
  --input path/to/input.wav \
  --output path/to/output.wav \
  --speaker-id 0
```

## References

Implementation is aligned in spirit with:

- `bshall/acoustic-model` for the SoftVC decoder and upsample recipe.
- `bshall/hubert` for content units.
- RMVPE for pitch extraction.
- `bshall/hifigan` for vocoding.

Bibliography:

- van Niekerk et al. A Comparison of Discrete and Soft Speech Units for Improved Voice Conversion. ICASSP 2022.
- Hsu et al. HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units. TASLP 2021.
- Qian et al. ContentVec: An Improved Self-Supervised Speech Representation by Disentangling Speakers. ICML 2022.
- Babu et al. XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale. Interspeech 2022.
- Pratap et al. Scaling Speech Technology to 1,000+ Languages. arXiv 2023.
- Wei et al. RMVPE: A Robust Model for Vocal Pitch Estimation in Polyphonic Music. Interspeech 2023.
- Kong et al. HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis. NeurIPS 2020.
- Défossez et al. Hybrid Spectrogram and Waveform Source Separation. ISMIR 2021.
- Wang et al. Music Source Separation with Band-Split RoPE Transformer. arXiv 2023.
- Wang et al. Neural Source-Filter-Based Waveform Model for Statistical Parametric Speech Synthesis. ICASSP 2019.
- Yamamoto et al. Parallel WaveGAN. ICASSP 2020.
