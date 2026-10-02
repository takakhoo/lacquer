# Revival notes (October 2026)

Why the thesis project (github.com/takakhoo/neural-audio-restoration) could not make tracks sound
better, and the design that replaced it. The thesis code stays in its own repo, unchanged; paths like
`Curriculum_Tokenize_Master/` below refer to that repo.

## 1. Why the thesis pipeline could not improve audio

Measured with `python -m remaster.diagnose_legacy` using the original functions and parameter
ranges from `Curriculum_Tokenize_Master/demastering.py`. Raw numbers: `docs/evidence/legacy/legacy_findings.json`.

| Finding | Evidence |
|---|---|
| The "reverb" was a boxcar lowpass with huge gain. `exp(-decay * t)` with decay 0.2-0.4 per second over a 50-150 ms kernel barely decays, so it sums ~100 ms of audio. | +72.8 dB DC gain, peaks grow 62x (median), energy above 1 kHz drops 31 dB. Written through `sf.write`, every reverb example was hard-clipped. Plot: `docs/evidence/legacy/legacy_reverb_ir.png` |
| The "EQ" was a band-pass. `scipy.signal.iirpeak` returns a resonator that passes only the band around `fc`; it does not boost or cut a band of an otherwise flat response. | -20 dB at 100 Hz and -21.6 dB at 10 kHz for fc = 1 kHz, Q = 1 |
| Gain was invisible to the model. The 48 kHz EnCodec normalizes each chunk and stores level in a separate scale value. | 100% of tokens unchanged after a +1 dB gain change |
| The codec is a quality ceiling below the input. A perfect token prediction decodes to the EnCodec reconstruction of the clean track. | Clean round trip: 9.6 dB SI-SDR, 10.3 dB log-spectral distance. The legacy echo-degraded input sits at 12.2 dB SI-SDR, closer to clean than a perfect model output could be |
| Audio-domain losses carried no gradient. `compute_loss` decodes `logits.argmax()`, which is not differentiable, and the mel/STFT terms compare two precomputed constants. | `token_train.py` lines 553-590. Only token cross-entropy trained the network |
| "Compression" was a memoryless waveshaper (no attack/release), i.e. distortion. | `apply_compression` in `demastering.py` |
| PESQ and STOI are narrowband speech metrics and say little about 44.1 kHz music. | Reported tables in the thesis README |

Net effect: 1.08B parameters were trained by cross-entropy alone to predict 16 codebooks of
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

**Stage 2, deterministic mastering (`remaster/master.py`).** Tonal balance toward a corpus-median
third-octave curve (bounded, smoothed, linear-phase), ITU-R BS.1770 loudness normalization, and a
4x-oversampled lookahead true-peak limiter. Loudness and peak control have exact DSP solutions, so no
network is used for them.

**Degradations (`remaster/degrade.py`)** are generated on the fly, time-aligned with the clean target:
real room impulse responses (MIT IR Survey, 270 IRs) and synthetic band-decaying tails with RT60
0.25-2.5 s at direct-to-reverberant ratios of -6..12 dB; 1-5 tap echoes at 60-500 ms with optional
feedback and darker repeats; 1-4 band peaking/shelving EQ at +-12 dB; a feed-forward compressor with
attack/release; hard/soft clipping; white/pink noise at 20-45 dB SNR. Segments are degraded with 2 s
of leading context so tails from earlier audio land inside the training crop.

## 3. Experiments

See `docs/experiments.md` (appended as runs finish).

## 4. Running it

```bash
python -m remaster.app --ckpt remaster/checkpoints/best.pt   # web UI at http://127.0.0.1:7860
python -m remaster.train --data data/raw/fma_medium --rir data/raw/mit_ir --out runs/x --task artifact
python -m remaster.evaluate --ckpt runs/x/best.pt --data data/raw/fma_medium --rir data/raw/mit_ir --out eval/x --task artifact
```

Training runs on a remote GPU box through `scripts/lab`, `scripts/sync`, `scripts/train_lab.sh`.

## 5. Deferred deliverable: visual walkthrough

Requested by Taka: once training curves are final and the system audibly works, produce a cohesive
set of figures in the spirit of `thesis_images/` (unet.png, encodec.png, structure.png, curriculum.png,
degradation_stack.png): the full signal path with real tensor shapes at every stage, the band-split
transformer internals, where EnCodec fits (analysis, see experiments), the decision flow
(echo -> room reverb -> vocal ambience -> VU riding -> mastering), and training curves.
Do this last, from measured shapes and logged curves, not from memory.
