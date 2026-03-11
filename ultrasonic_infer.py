#!/usr/bin/env python3
"""
Ultrasonic UV Panel Inference — Raspberry Pi 4

Classifies 1-second WAV segments from the Ultramic 384K EVO using a
TorchScript model exported from the training notebook.

Usage:
    # Single file
    python ultrasonic_infer.py --model ultrasonic_model.pt --wav test.wav

    # Watch a directory for new WAV files (continuous monitoring)
    python ultrasonic_infer.py --model ultrasonic_model.pt --watch /path/to/incoming/

    # Custom threshold
    python ultrasonic_infer.py --model ultrasonic_model.pt --wav test.wav --threshold 0.85
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import soundfile as sf
from PIL import Image


def load_wav(fn):
    """Load WAV file → (waveform [1, T], sample_rate)."""
    data, sr = sf.read(str(fn), always_2d=False)
    if data.ndim == 1:
        wav = torch.from_numpy(data).unsqueeze(0)
    else:
        wav = torch.from_numpy(data).T
        wav = wav.mean(dim=0, keepdim=True)
    return wav.float(), sr


def preprocess(wav, sr, meta):
    """
    Convert raw waveform to a 3-channel spectrogram tensor
    matching the training pipeline exactly.
    """
    sample_rate = meta['sample_rate']
    segment_sec = meta['segment_sec']
    n_fft = meta['n_fft']
    win_length = meta['win_length']
    hop_length = meta['hop_length']
    img_size = meta['img_size']

    # Resample if the recording sample rate differs from training
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)

    # Crop / pad to fixed length (center-crop if longer)
    num_samples = int(sample_rate * segment_sec)
    T = wav.shape[1]
    if T < num_samples:
        wav = F.pad(wav, (0, num_samples - T))
    elif T > num_samples:
        start = (T - num_samples) // 2
        wav = wav[:, start:start + num_samples]

    # Linear spectrogram (NOT mel — linear preserves ultrasonic detail)
    spec_fn = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        power=2.0,
    )
    to_db = torchaudio.transforms.AmplitudeToDB(stype='power')

    spec = spec_fn(wav)
    spec_db = to_db(spec)

    # Quantile normalization — must match training exactly
    lo = torch.quantile(spec_db, 0.10)
    hi = torch.quantile(spec_db, 0.99)
    spec_db = spec_db.clamp(lo, hi)
    spec_img = (spec_db - lo) / (hi - lo + 1e-9)

    # 3-channel image, resize to model input size
    img = spec_img.repeat(3, 1, 1).unsqueeze(0)
    img = F.interpolate(
        img, size=(img_size, img_size),
        mode='bilinear', align_corners=False
    )

    return img.squeeze(0), spec_img.squeeze(0)


def save_spectrogram_png(spec_1ch, out_path, img_size):
    """Save a single-channel spectrogram as a grayscale PNG."""
    x = spec_1ch.detach().cpu().numpy()
    x = (x - x.min()) / (x.max() - x.min() + 1e-9)
    x = (x * 255).astype(np.uint8)
    im = Image.fromarray(x)
    im = im.resize((img_size, img_size))
    im.save(str(out_path))


def predict_single(model, meta, wav_path, threshold, out_dir):
    """Run prediction on a single WAV file."""
    classes = meta['classes']
    wav, sr = load_wav(wav_path)
    img, spec_1ch = preprocess(wav, sr, meta)

    with torch.no_grad():
        logits = model(img.unsqueeze(0))
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    pred_idx = int(np.argmax(probs))
    pred_class = classes[pred_idx]
    confidence = float(probs[pred_idx])

    is_abnormal = pred_class != 'normal' and confidence >= threshold

    png_path = None
    if is_abnormal and out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time() * 1000)
        png_path = str(out_dir / f'{pred_class}_{ts}.png')
        save_spectrogram_png(spec_1ch, png_path, meta['img_size'])

    return {
        'wav': str(wav_path),
        'prediction': pred_class,
        'confidence': confidence,
        'probabilities': {c: float(probs[i]) for i, c in enumerate(classes)},
        'is_abnormal': is_abnormal,
        'spectrogram_png': png_path,
    }


def watch_directory(model, meta, watch_dir, threshold, out_dir, poll_sec=1.0):
    """Continuously monitor a directory for new WAV files."""
    watch_dir = Path(watch_dir)
    processed = set()

    print(f'Watching {watch_dir} for new .wav files (poll every {poll_sec}s)...')
    print(f'Threshold: {threshold} | Event output: {out_dir}')
    print('Press Ctrl+C to stop.\n')

    try:
        while True:
            wav_files = set(watch_dir.glob('*.wav'))
            new_files = wav_files - processed

            for wav_path in sorted(new_files):
                result = predict_single(model, meta, wav_path, threshold, out_dir)
                processed.add(wav_path)

                tag = 'ABNORMAL' if result['is_abnormal'] else 'OK'
                print(f'[{tag}] {wav_path.name}: '
                      f'{result["prediction"]} ({result["confidence"]:.1%})')

                if result['is_abnormal']:
                    print(json.dumps(result, indent=2))

            time.sleep(poll_sec)
    except KeyboardInterrupt:
        print(f'\nStopped. Processed {len(processed)} files total.')


def main():
    ap = argparse.ArgumentParser(
        description='Ultrasonic UV Panel Inference (Raspberry Pi 4)'
    )
    ap.add_argument('--model', required=True,
                    help='Path to TorchScript model (.pt)')
    ap.add_argument('--meta', default=None,
                    help='Path to meta.json (default: same folder as model)')
    ap.add_argument('--wav', default=None,
                    help='Single WAV file to classify')
    ap.add_argument('--watch', default=None,
                    help='Directory to watch for new WAV files')
    ap.add_argument('--threshold', type=float, default=0.80,
                    help='Confidence threshold for abnormal detection (default: 0.80)')
    ap.add_argument('--out_dir', default='events',
                    help='Directory for abnormal event spectrogram PNGs')
    args = ap.parse_args()

    if not args.wav and not args.watch:
        ap.error('Provide either --wav (single file) or --watch (directory)')

    # Load model + metadata
    model_path = Path(args.model)
    meta_path = Path(args.meta) if args.meta else model_path.parent / 'meta.json'

    if not meta_path.exists():
        print(f'Error: meta.json not found at {meta_path}', file=sys.stderr)
        sys.exit(1)

    meta = json.loads(meta_path.read_text())
    model = torch.jit.load(str(model_path), map_location='cpu').eval()

    print(f'Model : {model_path.name}')
    print(f'Classes: {meta["classes"]}')
    print(f'Sample rate: {meta["sample_rate"]:,} Hz')
    print(f'Freq range : 0 – {meta["sample_rate"] // 2:,} Hz')
    print()

    if args.wav:
        result = predict_single(
            model, meta, Path(args.wav), args.threshold, args.out_dir
        )
        print(json.dumps(result, indent=2))
    elif args.watch:
        watch_directory(
            model, meta, args.watch, args.threshold, args.out_dir
        )


if __name__ == '__main__':
    main()
