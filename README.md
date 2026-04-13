# Voice Cloning AI — From Scratch

A complete text-to-speech and voice cloning system built from scratch in PyTorch. Every component — the mel spectrogram pipeline, text frontend, speaker encoder, acoustic model, and vocoder — is implemented and documented without relying on high-level TTS libraries.

---

## What This Is

This project implements the full neural TTS pipeline used in modern voice cloning systems like VITS, Coqui TTS, and ElevenLabs — built from first principles across 6 progressive notebooks.

The goal was to understand every moving part deeply enough to explain it, train it, and debug it. Not to use a library. Not to fine-tune a black box.

---

## Architecture

```
Text Input
    │
    ▼
[Week 2 — Text Frontend]
    Normalization → Phonemization → ARPAbet tokenization
    │
    ▼
[Week 3 — Speaker Encoder]
    3-layer LSTM → 256-dim speaker embedding
    Trained with GE2E loss (leave-one-out centroids)
    │
    ▼
[Week 4 — Acoustic Model (FastSpeech2)]
    Token embedding + Speaker conditioning
    → 4× Transformer encoder blocks
    → Variance adaptor (duration, pitch, energy predictors)
    → Length regulator (phoneme → mel frame expansion)
    → 4× Transformer decoder blocks
    → Linear projection → Mel spectrogram [80 × T]
    │
    ▼
[Week 5 — Vocoder (HiFi-GAN)]
    4× Transposed conv upsample stages (8×8×2×2 = 256×)
    Multi-Receptive Field Fusion (3 parallel ResBlocks per stage)
    Multi-Period Discriminator (periods: 2, 3, 5, 7, 11)
    Multi-Scale Discriminator (full, 2×, 4× downsampled)
    │
    ▼
Audio Waveform
```

---

## Notebooks

| Notebook | What it builds |
|---|---|
| `week1_audio.ipynb` | Audio fundamentals — waveforms, STFT, mel filterbank, log mel spectrograms |
| `week2_audio.ipynb` | Text frontend — normalization, IPA phonemization, ARPAbet tokenization, padding, batching |
| `week3_audio.ipynb` | Speaker encoder — SpeakerMelSpectrogram, LSTM encoder, GE2E loss, speaker embeddings |
| `week4_audio.ipynb` | Acoustic model — PositionalEncoding, TransformerBlock, VariancePredictor, LengthRegulator, FastSpeech2 |
| `week5_audio.ipynb` | Vocoder — ResBlock, HiFi-GAN generator, MPD, MSD, GAN loss functions |
| `week6_audio.ipynb` | Training loop — overfit demo on single sentence, loss curves, audio output, checkpoint saving |

---

## Key Concepts Implemented

**Audio Processing (Week 1)**
- Short-Time Fourier Transform with Hann windowing
- Triangular mel filterbank construction from scratch (no librosa)
- Log mel spectrogram pipeline matching the 22050 Hz / 80-bin TTS standard

**Text Frontend (Week 2)**
- Text normalization: numbers, currency, abbreviations, symbols
- IPA phonemization via espeak-ng
- IPA → ARPAbet mapping (39-symbol American English phoneme set)
- Padding masks for transformer attention

**Speaker Encoder (Week 3)**
- 16kHz / 40-bin mel spectrogram for speaker ID
- 3-layer LSTM (768 hidden) → 256-dim L2-normalized embedding
- GE2E loss with leave-one-out centroid computation
- Windowed utterance embedding for arbitrary-length audio

**Acoustic Model (Week 4)**
- Sinusoidal positional encoding
- Multi-head self-attention with Pre-LN transformer blocks
- Duration, pitch, and energy variance predictors (Conv1d based)
- Length regulator: repeat-interleave phoneme expansion
- Pitch/energy quantization into learned embedding bins
- MSE loss on mel, duration, pitch, energy simultaneously

**Vocoder (Week 5)**
- Dilated residual blocks with weight normalization
- Multi-Receptive Field Fusion: parallel ResBlocks at kernels [3, 7, 11]
- ConvTranspose1d upsampling (8×8×2×2 = 256× total)
- Least-squares GAN loss (LSGAN) for stable training
- Feature matching loss for perceptual quality
- Mel reconstruction loss (weight=45) for content fidelity

**Training (Week 6)**
- Full end-to-end training loop connecting all components
- Separate AdamW optimizers for acoustic model, generator, discriminators
- Gradient clipping for transformer stability
- Checkpoint saving and resuming
- Overfit demo: acoustic mel loss 59.7 → 0.84 in 500 steps

---

## Results

Training on a single sentence ("hello my name is aryan") for 500 steps on CPU:

```
Step    1: Acoustic loss = 59.71  (mel = 54.06)
Step  100: Acoustic loss =  6.71  (mel =  6.65)
Step  250: Acoustic loss =  1.22  (mel =  1.18)
Step  500: Acoustic loss =  0.84  (mel =  0.82)

Mel loss reduction: 98.5% in 500 steps
```

Audio output via Griffin-Lim reconstruction shows correct phoneme rhythm and pitch contour. Full audio quality requires HiFi-GAN training for 500k+ steps on GPU.

---

## Setup

```bash
# Clone the repo
git clone https://github.com/Srik-007/Voice_Cloning-AI-Model.git
cd Voice_Cloning-AI-Model

# Create venv
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# System dependencies (Arch Linux)
sudo pacman -S espeak-ng ffmpeg

# Launch notebooks
jupyter notebook
```

Run notebooks in order: week1 → week2 → week3 → week4 → week5 → week6.

---

## Dependencies

- Python 3.11+
- PyTorch 2.x + torchaudio
- espeak-ng (system package, for phonemization)
- phonemizer, inflect, unidecode (text frontend)
- scikit-learn (PCA visualization in Week 3)
- soundfile (audio saving)
- See `requirements.txt` for full list

---

## Project Structure

```
Voice_Cloning-AI-Model/
├── week1_audio.ipynb    # Audio pipeline
├── week2_audio.ipynb    # Text frontend
├── week3_audio.ipynb    # Speaker encoder
├── week4_audio.ipynb    # FastSpeech2 acoustic model
├── week5_audio.ipynb    # HiFi-GAN vocoder
├── week6_audio.ipynb    # Training loop + demo
├── requirements.txt
└── README.md
```

---

## What's Next

- [ ] Train on real voice recordings (requires GPU — Google Colab recommended)
- [ ] Montreal Forced Aligner for accurate phoneme duration labels
- [ ] MeloTTS voice cloning demo with reference audio
- [ ] FastAPI backend for inference
- [ ] Web frontend with waveform visualization

---

## References

- [FastSpeech 2](https://arxiv.org/abs/2006.04558) — Ren et al. 2020
- [HiFi-GAN](https://arxiv.org/abs/2010.05646) — Kong et al. 2020
- [Generalized End-to-End Loss (GE2E)](https://arxiv.org/abs/1710.10467) — Wan et al. 2018
- [VITS](https://arxiv.org/abs/2106.06103) — Kim et al. 2021
- [Transfer Learning from Speaker Verification (SV2TTS)](https://arxiv.org/abs/1806.04558) — Jia et al. 2018
