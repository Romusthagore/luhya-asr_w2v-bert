# Luhya ASR — Automatic Speech Recognition for the Luhya Language

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.30+-yellow.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Training](#training)
- [Results](#results)
- [Usage](#usage)
- [Metrics](#metrics)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project implements an Automatic Speech Recognition (ASR) system for the Luhya language, built on Facebook's pre-trained Wav2Vec2-BERT 2.0 model.

The model was fine-tuned on 10 hours of audio data and achieves a Word Error Rate (WER) of 54.63% and a Character Error Rate (CER) of 12.67%.

### Key Results

| Metric | Result |
|--------|--------|
| WER | 54.63% |
| CER | 12.67% |
| Loss | 0.665 |
| Score | 66.35 |
| Training Time | 2h50 |

---

## Features

- Complete ASR model for the Luhya language
- Fine-tuning of Wav2Vec2-BERT 2.0 (580M parameters)
- Robust data pipeline with filtering and preprocessing
- Evaluation metrics: WER, CER, Score
- Automatic checkpointing and saving
- Colab support optimized for T4 GPU
- Modular, well-structured codebase

---

## Project Structure

```
luhya-asr_w2v-bert/
├── configs/
│   └── train_config_colab.yaml     # Training configuration
├── scripts/
│   └── train_model.py              # Main training script
├── src/
│   ├── data/
│   │   ├── preprocessing.py        # Text cleaning
│   │   ├── dataset.py              # Loading and filtering
│   │   └── dataset_encoders.py     # Encoding for training
│   ├── models/
│   │   ├── factory.py              # Model creation
│   │   └── hubert_with_adapter.py  # Hubert with adapters
│   ├── training/
│   │   ├── collator.py             # Dynamic padding
│   │   ├── metrics.py              # WER, CER, Score
│   │   └── trainer.py              # Trainer configuration
│   └── utils/
│       ├── config.py               # Central configuration
│       └── cache.py                # Dataset caching
├── notebooks/
│   └── training.ipynb              # Training notebook
├── bash_scripts/
│   └── train.sh                    # Launch script
├── requirements.txt                # Dependencies
└── README.md                       # Documentation
```

---

## Installation

### Prerequisites

- Python 3.10 or later
- CUDA-compatible GPU (recommended)
- Google Colab (optional)

### Installing Dependencies

```bash
# Clone the repository
git clone https://github.com/Romusthagore/luhya-asr_w2v-bert.git
cd luhya-asr_w2v-bert

# Install dependencies
pip install -r requirements.txt
```

### Main Dependencies

```txt
torch>=2.0.0
transformers>=4.30.0
datasets>=2.10.0
evaluate>=0.4.0
jiwer>=3.0.0
pyarrow>=12.0.0
tqdm>=4.65.0
```

---

## Configuration

Configuration is defined in `configs/train_config_colab.yaml`:

```yaml
# Model
pretrained_model: "w2v-BERT"           # Wav2Vec2-BERT 2.0
freeze_feature_encoder: true
add_final_layer_adapter: true

# Training
batch_size: 4
gradient_accumulation_steps: 8
num_epochs: 2
learning_rate: 5e-5
warmup_steps: 500
fp16: true

# Data
dataset_path: "DDD-Kenya/Luhya-ASR-Data-subset-50h"
max_data_hours: 10.0
validation_split_pct: 0.2
```

---

## Training

### On Colab

```bash
python3 scripts/train_model.py --config configs/train_config_colab.yaml
```

### Via Bash Script

```bash
chmod +x bash_scripts/train.sh
./bash_scripts/train.sh
```

### Resuming from a Checkpoint

```bash
python3 scripts/train_model.py \
    --config configs/train_config_colab.yaml \
    --resume_from_checkpoint results/experiment/checkpoint-xxx
```

---

## Results

### Final Metrics

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| WER   | 54.63% | < 60%  | Met |
| CER   | 12.67% | < 20%  | Met |
| Loss  | 0.665  | —      | — |
| Score | 66.35  | > 60   | Met |

### Training Progression

| Step | WER | CER | Loss | Score |
|------|-----|-----|------|-------|
| Step 50  | 100.00% | 94.39% | 3.345 | 2.80 |
| Step 100 | 98.92%  | 40.98% | 1.581 | 30.05 |
| Step 150 | 68.38%  | 19.82% | 0.920 | 55.90 |
| Step 200 | 55.74%  | 12.90% | 0.674 | 65.68 |
| Final    | 54.63%  | 12.67% | 0.665 | 66.35 |

### Sample Prediction

```
Prediction: ramo ichumbe mumaenjera mana khande osasie.
Reference:  ramo ichumbe mumaenjera mana khandi osasie.
WER: 16.67% | CER: 2.33%
```

### Visualization

![Performance Metrics](performance_metrics.png)

---

## Usage

### Loading the Model

```python
from transformers import Wav2Vec2BertForCTC, Wav2Vec2BertProcessor
import torch

# Load model
model_path = "path/to/final_model"
processor_path = "path/to/processor"

model = Wav2Vec2BertForCTC.from_pretrained(model_path)
processor = Wav2Vec2BertProcessor.from_pretrained(processor_path)

# Set to evaluation mode
model.eval()
```

### Transcribing Audio

```python
import torchaudio

# Load audio
audio, sr = torchaudio.load("audio.wav")

# Resample to 16000 Hz if needed
if sr != 16000:
    resampler = torchaudio.transforms.Resample(sr, 16000)
    audio = resampler(audio)

# Extract features
inputs = processor(audio.numpy(), sampling_rate=16000, return_tensors="pt")

# Prediction
with torch.no_grad():
    logits = model(inputs.input_values).logits

# Decode
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)

print(f"Transcription: {transcription[0]}")
```

---

## Metrics

### WER — Word Error Rate

Measures the error rate at the word level:

```
WER = (Substitutions + Insertions + Deletions) / Total number of words
```

### CER — Character Error Rate

Measures the error rate at the character level:

```
CER = (Substitutions + Insertions + Deletions) / Total number of characters
```

### Score

Combines WER and CER into a single indicator:

```
Score = (1 - (WER + CER) / 2) x 100
```

---

## Contributing

Contributions are welcome. To contribute:

1. Fork the repository
2. Create a branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Hugging Face, for the `transformers` and `datasets` libraries
- Facebook AI, for the Wav2Vec2-BERT 2.0 model
- DDD-Kenya, for the Luhya dataset

---

