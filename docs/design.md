# Design notes

Why Lacquer corrects a spectrogram instead of generating audio from codec tokens, and how the stages are
split. Section 1 measures a token-prediction pipeline I built first: EnCodec tokens in, a 1-D U-Net trained
with cross-entropy, EnCodec tokens out.

## 1. Why token prediction could not improve audio

Measured with `python -m remaster.diagnose_legacy`, which re-implements that pipeline's degradations and
parameter ranges. Raw numbers: `docs/evidence/legacy/legacy_findings.json`.

| Finding | Evidence |
|---|---|
| The "reverb" was a boxcar lowpass with huge gain. `exp(-decay * t)` with decay 0.2-0.4 per second over a 50-150 ms kernel barely decays, so it sums ~100 ms of audio. | +72.8 dB DC gain, peaks grow 62x (median), energy above 1 kHz drops 31 dB. Written through `sf.write`, every reverb example was hard-clipped. Plot: `docs/evidence/legacy/legacy_reverb_ir.png` |
| The "EQ" was a band-pass. `scipy.signal.iirpeak` returns a resonator that passes only the band around `fc`; it does not boost or cut a band of an otherwise flat response. | -20 dB at 100 Hz and -21.6 dB at 10 kHz for fc = 1 kHz, Q = 1 |
| Gain was invisible to the model. The 48 kHz EnCodec normalizes each chunk and stores level in a separate scale value. | 100% of tokens unchanged after a +1 dB gain change |
| The codec is a quality ceiling below the input. A perfect token prediction decodes to the EnCodec reconstruction of the clean track. | Clean round trip: 9.6 dB SI-SDR, 10.3 dB log-spectral distance. The legacy echo-degraded input sits at 12.2 dB SI-SDR, closer to clean than a perfect model output could be |
| Audio-domain losses carried no gradient. `compute_loss` decodes `logits.argmax()`, which is not differentiable, and the mel/STFT terms compare two precomputed constants. | The old training loop. Only token cross-entropy trained the network |
| "Compression" was a memoryless waveshaper (no attack/release), i.e. distortion. | the old compression function |
| PESQ and STOI are narrowband speech metrics and say little about 44.1 kHz music. | They were the reported metrics |

Net effect: 1.08 B parameters were trained by cross-entropy alone to predict 16 codebooks of
mostly noise-like residual tokens, on degradations that were either inaudible (gain, +-1 dB), invisible
in token space (gain), or mislabeled (reverb, EQ), with an output ceiling lower than the input quality.

## 2. New design

Two stages, split by whether the problem has a unique right answer.

**Stage 1, neural restoration (`remaster/model.py`).** Band-split transformer (BS-RoFormer family)
over the stereo complex STFT at 44.1 kHz. It predicts a complex mask that multiplies the input
spectrogram, and the mask head is zero-initialized, so the untrained network is an exact identity.
There is no codec or vocoder in the path: anything the model does not touch passes through bit-exact,
which removes the quality ceiling. Differences from stock BS-RoFormer:

- per-band log-level features are concatenated to the normalized band features. Stock band-split
  normalizes level away, which is fine for source separation but discards exactly what reverb decay,
  EQ and compression change.
- loss = waveform L1 + multi-resolution complex STFT L1 + log-magnitude L1. The log term is there
  because reverb tails and echo repeats are quiet and nearly invisible to linear-domain losses.

**Mastering (`remaster/analysis.py`, `remaster/mastering.py`, `remaster/stem_master.py`).** See section 3.

**Degradations (`remaster/degrade.py`)** are generated on the fly, time-aligned with the clean target:
real room impulse responses (MIT IR Survey, 270 IRs) and synthetic band-decaying tails with RT60
0.25-2.5 s at direct-to-reverberant ratios of -6..12 dB; 1-5 tap echoes at 60-500 ms with optional
feedback and darker repeats; 1-4 band peaking/shelving EQ at +-12 dB; a feed-forward compressor with
attack/release; hard/soft clipping; white/pink noise at 20-45 dB SNR. Segments are degraded with 2 s
of leading context so tails from earlier audio land inside the training crop.

## 3. Why the mastering half is measurement plus small moves

Mastering is the part where a network is least justified, and the measurements say why.

**A good mix already has the right tone.** Unmastered professional mixes (MUSDB18-HQ) and 103,838 released
tracks have nearly the same spectral slope (-5.1 against -5.3 dB/octave) and stereo image, and differ by 4 dB of peak-to-loudness
ratio (E24). What mastering adds to a good mix is mostly dynamics control and loudness, which have exact DSP
formulations and published delivery specs.

**A bad tone cannot be identified blind.** A track's own spectrum is 5.2 dB RMS from the population mean and
4.3 dB from the mean of its nearest neighbours, while a typical tonal fault is 2.3 dB (E25). Whatever the
estimator (range rule, Gaussian posterior mean, genre conditioning, the learned controller of E14), the fault
is smaller than the natural spread it has to be told apart from. A reference-free quality predictor does not
see these faults either (E26). So the chain does what mastering engineers describe: moves of at most 1.5 dB
without a reference, a note in the report when the reading is further out than mastering should fix, and full
matching only when a reference track says what the target is.

**Every action has a budget and a reading.** The limiter may take 3 dB; a loudness target that would need more
is not reached and the report says so. The true-peak ceiling tightens to -2 dBTP for loud masters. Nothing is
filtered as routine. Each stage reports the measured value, the normal range it was compared with, and what it
did.

**Stems are for balance.** Instrument tone varies more between songs than a typical fault, so pulling a stem
toward the average of its kind made mixes worse in the first stem evaluation. Stem loudness relative to the mix
is tighter, so that is the one stem-level correction, measured on separated stems against norms measured the
same way, and applied as a difference so that untouched stems add nothing.

**The limiter is engineered and measured.** A slow stage carries sustained reduction, a fast lookahead stage
catches the rest, and an oversampled soft clipper takes the last 1 to 2 dB off the shortest peaks. It is
compared with other limiters at equal loudness and equal true peak on two signal measures, distortion and gain
movement (E28, E29).

## 4. Experiments

See `docs/experiments.md` (appended as runs finish).

## 5. Running it

```bash
python -m remaster.app --ckpt remaster/checkpoints/best.pt   # web UI at http://127.0.0.1:7860
python -m remaster.train --data data/raw/fma_medium --rir data/raw/mit_ir --out runs/x --task artifact
python -m remaster.evaluate --ckpt runs/x/best.pt --data data/raw/fma_medium --rir data/raw/mit_ir --out eval/x --task artifact
```

Training runs on a remote GPU box through `scripts/lab`, `scripts/sync`, `scripts/train_lab.sh`.
