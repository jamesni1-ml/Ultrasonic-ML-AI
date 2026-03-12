# Ultrasonic-ML-AI

Ultrasonic monitoring system for UV panel fault detection using deep learning on Raspberry Pi 4.

## Overview

This project uses a **[Dodotronic Ultramic 384K EVO](https://www.dodotronic.com/product/ultramic-384k-evo/)** USB ultrasonic microphone to detect anomalies in UV panel operation. Audio captured at **384 kHz** (frequency range 0–192 kHz) is converted to **linear spectrograms** and classified using a **MobileNetV2** CNN, optimized for real-time inference on Raspberry Pi 4.

### Why these choices?

| Decision | Reasoning |
|----------|-----------|
| **Linear spectrogram** (not mel) | Mel scale is tuned for human hearing (0–20 kHz). Ultrasonic signals above 20 kHz need linear frequency spacing to preserve detail. |
| **MobileNetV2** (not ResNet18) | ~3.4M params vs ~11.7M. Purpose-built for edge/mobile devices. Runs efficiently on Pi 4's ARM Cortex-A72. |
| **384 kHz sample rate** | Matches the Ultramic 384K EVO's native rate. Captures the full 0–192 kHz range (Nyquist). |
| **TorchScript export** | No fastai dependency needed at inference time. Smaller deployment footprint on the Pi. |

## Features

- **Full ultrasonic spectrum** — 0 to 192 kHz via 384 kHz sampling
- **Linear spectrogram pipeline** — preserves high-frequency detail that mel suppresses
- **MobileNetV2 classifier** — lightweight CNN for Pi 4 edge deployment (~3.4M parameters)
- **Multi-class detection** — normal, abnormal, abnormal2, pulsing_unstable (extensible)
- **TorchScript export** — standalone inference with no fastai dependency
- **Two inference modes** — single file classification or continuous directory monitoring
- **Event logging** — saves spectrogram PNGs and JSON results for abnormal detections

## Hardware

| Component | Model | Notes |
|-----------|-------|-------|
| Microphone | [Dodotronic Ultramic 384K EVO](https://www.dodotronic.com/product/ultramic-384k-evo/) | USB UAC 1.1 — driverless on Linux |
| Edge Device | Raspberry Pi 4 (2 GB+ RAM) | Runs inference in real time |
| Training Machine | Any machine with CUDA GPU | Recommended for faster training |

## Project Structure

```
Ultrasonic-ML-AI/
├── README.md
├── .gitignore
├── requirements.txt            # Training dependencies (GPU workstation)
├── requirements-pi.txt         # Inference dependencies (Raspberry Pi 4)
├── ultrasonic_training.ipynb   # Training notebook (Jupyter)
├── ultrasonic_infer.py         # Inference script for Pi 4
└── node-red/                   # Node-RED integration
    ├── ultrasonic-classifier.js    # Custom node (JS)
    ├── ultrasonic-classifier.html  # Node editor UI
    ├── package.json                # Node-RED package
    ├── README.md                   # Node-RED setup guide
    └── examples/                   # Ready-to-import flows
        ├── flow-mqtt.json
        ├── flow-tcp.json
        ├── flow-bluetooth.json
        └── flow-all-outputs.json
```

After training, the notebook exports:
```
├── ultrasonic_model.pt         # TorchScript model (deploy this)
├── meta.json                   # Model config (deploy this too)
└── ultrasonic_model_fastai.pkl # fastai export (optional backup)
```

## Data Preparation

Organize your WAV recordings into folders by class:

```
data/ultrasonic/
├── normal/
│   ├── clip_001.wav
│   └── ...
├── abnormal/
│   ├── clip_001.wav
│   └── ...
├── abnormal2/
│   └── ...
└── pulsing_unstable/
    └── ...
```

- Record at **384 kHz** using the Ultramic 384K EVO
- Segment recordings into **1-second clips** (384,000 samples each)
- Place each clip in the folder matching its class label

## Quick Start

### Training (workstation with GPU)

```bash
pip install -r requirements.txt
```

1. Place your data in `data/ultrasonic/` as shown above
2. Open `ultrasonic_training.ipynb` in Jupyter
3. Update `PROJECT_ROOT` and `DATA_ROOT` paths in Cell 3
4. Run all cells — the notebook will:
   - Convert WAVs to linear spectrograms
   - Train a MobileNetV2 classifier with transfer learning
   - Export `ultrasonic_model.pt` + `meta.json`

### Inference (Raspberry Pi 4)

```bash
pip install -r requirements-pi.txt
```

Copy `ultrasonic_model.pt` and `meta.json` to the Pi, then:

```bash
# Classify a single file
python ultrasonic_infer.py --model ultrasonic_model.pt --wav recording.wav

# Watch a directory for new files (continuous monitoring)
python ultrasonic_infer.py --model ultrasonic_model.pt --watch /path/to/incoming/

# Custom confidence threshold
python ultrasonic_infer.py --model ultrasonic_model.pt --wav recording.wav --threshold 0.85
```

Abnormal detections automatically save a spectrogram PNG to the `events/` folder.

## Model Specifications

| Parameter | Value |
|-----------|-------|
| Architecture | MobileNetV2 (pretrained on ImageNet) |
| Input size | 224 × 224 × 3 (linear spectrogram) |
| Sample rate | 384,000 Hz |
| Segment length | 1.0 second |
| FFT size (n_fft) | 4,096 |
| Window length | 2,048 |
| Hop length | 1,024 |
| Frequency range | 0 – 192,000 Hz |
| Spectrogram type | Linear power spectrogram (dB-scaled) |
| Normalization | Per-sample quantile (10th–99th percentile) |

## Microphone Setup on Raspberry Pi

The Ultramic 384K EVO is a standard USB audio class (UAC 1.1) device — **no drivers needed** on Raspberry Pi OS.

```bash
# Verify the mic is detected
arecord -l

# Record a 5-second test clip at 384 kHz
arecord -D hw:X,0 -f S16_LE -r 384000 -c 1 -d 5 test.wav
```
*(Replace `X` with your device number from `arecord -l`)*

## Node-RED Integration

A custom Node-RED node is included for visual, low-code deployment on the Pi.

### Output options

| Output | Protocol | Abnormal payload | Normal payload |
|--------|----------|-----------------|----------------|
| **MQTT** | MQTT publish | JSON + base64 spectrogram PNG | JSON status only |
| **TCP** | TCP socket | JSON + base64 spectrogram PNG | JSON status only |
| **Bluetooth** | BLE GATT | Compact JSON (no image — MTU limit) | Compact JSON |

### Quick setup

```bash
# Install Node-RED on Pi
bash <(curl -sL https://raw.githubusercontent.com/node-red/linux-installers/master/deb/update-nodejs-and-nodered)

# Install the custom node
cd ~/.node-red
npm install /path/to/Ultrasonic-ML-AI/node-red

# Restart
node-red-restart
```

Import an example flow from `node-red/examples/` and configure the model path.

See [node-red/README.md](node-red/README.md) for full details.

## License

MIT
