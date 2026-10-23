# Finetuning BirdNET  
**Framework for bioacoustic classification using BirdNET.**

A comprehensive solution for ornithological research combining **BirdNET fine-tuning**, **custom CNN architectures**, and **advanced audio preprocessing pipelines**.  

---

## 📘 Overview

This repository provides an end-to-end machine learning pipeline for bird sound classification, featuring:

- Dataset preprocessing with automated segmentation and annotation processing​
- Custom CNN training with configurable architectures and hyperparameters​
- BirdNET fine-tuning on domain-specific datasets with TensorFlow Lite optimization​
- Advanced data augmentation including pitch shift, time stretch, and background noise injection​
- Threshold optimization for multi-label classification performance​
- Comprehensive evaluation with precision, recall, F1-score (micro/macro/weighted/samples)
---

## ⚙️ Key Features

- **Multi-model support:** Train custom CNNs or fine-tune pre-trained BirdNET models  
- **Scalable architecture:** Batch processing, GPU acceleration, multi-threading support  
- **Scientific validation:** Per-class metrics, confusion matrices, optimal threshold computation  
- **Reproducibility:** Comprehensive logging, checkpointing, and configuration management

---

## 🏗️ Architecture

### Pipeline Components

The system consists of four main modules:

1. **Dataset Builder** (`build_dataset.py`)  
   Modular processor for audio segment extraction and dataset organization

2. **Preprocessing Pipeline** (`preprocessing.py`)  
   Integrated workflow for annotation processing, train/valid/test splitting, data augmentation, and spectrogram generation

3. **Custom CNN Training** (`custom_cnn_training.py`)  
   Training framework for custom Vanilla CNN architectures with PyTorch

4. **BirdNET Testing** (`birdnet_testing.py`)  
   Pipeline for BirdNET fine-tuning, inference, threshold optimization, and evaluation

---

## 🧩 Installation

### Prerequisites
- Python ≥ 3.8  
- CUDA-compatible GPU *(optional, for accelerated training)*  
- FFmpeg *(for audio processing)*

### Dependencies
```bash
pip install torch torchvision torchaudio
pip install click numpy pandas scikit-learn
pip install librosa soundfile pydub audiomentations
pip install matplotlib tqdm scipy
pip install birdnetlib  # For BirdNET integration
```

### BirdNET Analyzer Setup
```bash
git clone https://github.com/kahst/BirdNET-Analyzer.git
```
Place it in the **project root directory.**

---

## 🚀 Quick Start

### 1. Dataset Preparation
Convert and normalize audio files:
```bash
python build_dataset.py   \
  --audio-source /path/to/raw/audio   \
  --dataset-path segments   \
  --bird-tags-files /path/to/train/labels /path/to/test/labels   \
  --species-dict BirdNET_GLOBAL_6K_V2.4_Labels_en_uk.txt
```

Integrate annotated segments:
```bash
python preprocessing.py integrate-segments   
  --audio-source /path/to/raw/audio   \
  --annotation-file /path/to/train/labels   \
  --labels-file BirdNET_GLOBAL_6K_V2.4_Labels_en_uk.txt
```

### 2. Preprocessing & Augmentation
Split dataset:
```bash
python preprocessing.py preprocess   
  --segments-base segments   \
  --train-split 0.85   \
  --valid-split 0.15
```

Apply data augmentation:
```bash
python preprocessing.py augment   \
  --target-count 1000   \
  --max-none-samples 5000
```

Generate mel-spectrograms:
```bash
python preprocessing.py generate-spectrograms   \
  --dataset-variant test
```

### 3. Model Training
#### Option A — Custom CNN Training
```bash
python custom_cnn_training.py train   \
  --model-name CustomCNN   \
  --dataset-variant test   \
  --num-conv-layers 4   \
  --kernel-sizes 4,5,6,6  \
  --channels 16,32,64,128   \
  --batch-size 128   \
  --epochs 200   \
  --learning-rate 1e-4   \
  --gpu-id 0
```
Evaluate:
```bash
python custom_cnn_training.py evaluate   \
  --model-name CustomCNN   \
  --dataset-variant test
```

#### Option B — BirdNET Fine-tuning
```bash
python birdnet_testing.py finetune   \
  --dataset-variant test   \
  --batch-size 64   \
  --epochs 150   \
  --threads 16   \
  --mixup
```

### 4. Inference & Evaluation
Run BirdNET inference:
```bash
python birdnet_testing.py analyze   \
  --dataset-variant test   \
  --split both   \
  --min-conf 0.05   \
  --sensitivity 1.0
```
Optimize thresholds:
```bash
python birdnet_testing.py optimize-thresholds   \
  --dataset-variant test   \
  --num-thresholds 200
```
Evaluate:
```bash
python birdnet_testing.py evaluate   \
  --dataset-variant test
```
Full automated pipeline:
```bash
python birdnet_testing.py full-pipeline   \
  --dataset-variant test   \
  --batch-size 64   \
  --epochs 150   \
  --threads 16
```

---

## ⚙️ Configuration

### Custom CNN Architecture
- Convolutional layers: 1–8  
- Channels: configurable  
- Kernel sizes: 3×3 to 7×7  
- Dense layers: 64–512 units  
- Dropout: 0.0–0.5  
- Input shape: 256×256 mel-spectrograms

### Training Parameters
| Parameter | Range / Default |
|------------|----------------|
| Batch size | 32–256 |
| Learning rate | 1e-5 → 1e-3 |
| Optimizer | Adam |
| Early stopping | 10 epochs |
| LR scheduler | ReduceLROnPlateau |
| Augmentation | Mixup, pitch/time/noise |

---

## 🗂️ Dataset Configuration

**Structure:**
```
segments/
├── train/
│   ├── Species_1/
│   ├── Species_2/
│   └── None/
├── valid/
└── test/

utils/
└── dataset/
    ├── dataset_config_custom.json
    ├── true_segments_train.json
    └── audio_info.json
```

---

## 🎛️ Data Augmentation

### Methods
- Pitch Shift: ±3 semitones (psA, psB)  
- Time Stretch: 0.9×–1.1×  
- Background Noise: From “None” class  
- Gain Adjustment: ±5 dB  
- Mixup: BirdNET only

Example:
```bash
python preprocessing.py augment   --target-count 1000   --max-none-samples 5000   --augmentation-methods time_stretch pitch_shift add_noise
```

---

## 📈 Model Evaluation

### Metrics Computed
- Multi-label metrics: Micro / Macro / Weighted / Samples F1  
- Per-class: Precision, Recall, F1, Support  
- Confusion analysis  
- Threshold optimization

### Outputs

**Custom CNN:**
```
models/CustomCNN/custom/
├── checkpoint.pth
├── final_weights.pth
├── history_loss.npy
├── history_valid_loss.npy
├── training_history.png
└── evaluation_results.json
```

**BirdNET:**
```
models/BirdNET_tuned/test/
├── test.tflite
├── test_Labels.txt
├── optimized_thresholds.json
├── test_evaluation_report.csv
└── test_pred_segments.json
```

---

## 🧮 Utility Functions

### Dataset Statistics
```bash
python preprocessing.py stats   --output-csv dataset_statistics.csv
```

### Prepare Dataset Configuration
```bash
python preprocessing.py prepare-dataset   --dataset-variant custom   --force-regenerate
```

---

## 🔬 Advanced Usage

### Custom Model Architectures
`model_configs.json`
```json
[
  {
    "config": {
      "num_conv_layers": 6,
      "channels": [32, 64, 128, 256, 512, 512],
      "kernel_sizes": [5, 5, 5, 3, 3, 3],
      "dense_hidden": 256,
      "dropout": 0.3,
      "batch_size": 64
    }
  }
]
```
Use with:
```bash
python custom_cnn_training.py train   --config-file model_configs.json   --resume
```

### Multi-GPU Training
```bash
python custom_cnn_training.py train   --gpu-id 0   # Use first GPU
  --gpu-id 1   # Use second GPU
  --gpu-id -1  # Force CPU
```

### Batch Experiments
```bash
for variant in base_v1 augm_v1 augm_v2; do
  python birdnet_testing.py full-pipeline     --dataset-variant $variant     --epochs 150     --threads 16
done
```

---

## 📁 Project Structure
```
.
├── build_dataset.py
├── preprocessing.py
├── custom_cnn_training.py
├── birdnet_testing.py
├── birdlib/
│   └── utils.py
├── segments/
├── models/
├── utils/
└── BirdNET-Analyzer/
```

---
