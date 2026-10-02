# Experiment log

Newest last. Metrics are SI-SDR in dB on the fixed validation set (32 held-out FMA tracks, 6 s each, one item per condition), input -> output.

## 2026-10-01

### E0. Level features in the band split
Stock BS-RoFormer band split normalizes each band to unit norm. Added per-band log level as extra
input features (`BandSplit.level_feats`). Rationale and code: `remaster/model.py`. Both runs below use it.

### E1. Joint (undo everything) vs artifact specialist, from scratch
Same model (dim 128, depth 6, 7.7M params), batch 8, 4 s segments, lr 3e-4, one RTX 6000 Ada shared.

| run | step | reverb | echo | clip | noise | identity |
|---|---:|---|---|---|---|---|
| `small` (joint: reverb, echo, EQ, comp, clip, noise) | 6000 | 2.2 -> 3.1 | 9.5 -> 9.1 | 18.9 -> 17.7 | 33.1 -> 24.1 | 117 -> 26.8 |
| `art_small` (artifact: reverb, echo, clip, noise; EQ/comp as augmentation) | 2000 | 2.9 -> 3.7 | 9.5 -> 9.0 | 19.6 -> 17.2 | 31.0 -> 23.0 | 117 -> 25.0 |

Joint also: eq 9.9 -> 9.1, comp 17.3 -> 16.1. Both learn slowly from scratch: log-magnitude loss
falls steadily while SI-SDR barely moves, because a reverberant mix has scrambled phase that a
multiplicative mask cannot fully repair. Joint run stopped at 6k steps to free the GPU. The two
validation sets differ (artifact applies EQ/comp augmentation to the clean side), so read each
row as input vs output on its own set.

### E2. Public vocal dereverb RoFormers, zero-shot on full mixes
Checkpoints: anvuew `dereverb_bs_roformer` (51M params) and `dereverb_mel_band_roformer` (228M), GPL-3.0,
trained on vocals. Evaluated on the artifact validation set (SI-SDR dB, input -> output).

| model | reverb | echo | reverb+echo | identity |
|---|---|---|---|---|
| BS-RoFormer (vocal dereverb) | 2.9 -> -0.6 | 9.5 -> 0.9 | 0.9 -> -4.8 | 117 -> 5.3 |
| Mel-Band RoFormer (vocal dereverb) | 2.9 -> -16.7 | 9.5 -> -5.6 | 0.9 -> -18.5 | 117 -> -2.4 |

Vocal models damage full mixes badly, including clean input. Full-mix restoration needs its own training.

### E3. Fine-tuning the vocal BS-RoFormer on full-mix artifacts (running)
`ft_bs`: init from the anvuew BS-RoFormer, artifact task, batch 4, lr 1e-4. Results pending.

### E4. DSP de-echo (cepstral detection + exact inverse filter), no training
`remaster/deecho.py`. A delayed copy puts a ripple on the log spectrum, which is a narrow peak in the
cepstrum at the delay. The peak gives delay and echo path; the echo is inverted with a recursive
filter, refined by fixed-point iteration, and peeled repeatedly for multi-tap delays.
79 held-out FMA clips (25 s), echo from `degrade.apply_echo` (60-500 ms, gain 0.15-0.6, 1-5 taps,
optional lowpassed repeats):

| | SI-SDR in | SI-SDR out |
|---|---:|---:|
| all (n=79) | 8.8 | 25.2 (median 23.2) |
| single tap (n=48) | 9.4 | 29.2 |
| multi-tap / feedback (n=31) | 7.9 | 19.0 |

Detection 71/79 at z >= 45 (misses are the weakest echoes). On the same clips with no echo added it
fires on 3/79, with cepstral gains 0.12-0.44 at 361-522 ms, which look like tempo-synced delay or
exact loops already in the music. One echoed clip came out worse than its input (14.6 -> 6.9 dB).
Runtime 0.6 s per 25 s clip on CPU. For comparison the neural models so far leave echo at
9.5 -> 9.0 dB. Decision: discrete echo goes to this DSP stage, exposed as a toggle that reports what
it removed; the network keeps reverb, clipping and noise.

### E5. Can tonal-balance matching undo bad EQ? (mastering stage)
79 held-out clips, random 1-4 band EQ at up to +-12 dB. Metric: RMS third-octave spectrum distance
to the clean track (dB, 40 Hz-12 kHz, level-aligned). "Clean moved" is how far the same processing
pushes an undamaged track.

| tonal mode | EQ'd: in -> out | clean moved |
|---|---|---:|
| pull to corpus median, strength 1.0 | 2.62 -> 3.68 | 3.64 |
| pull to corpus median, strength 0.6 | 2.62 -> 2.77 | 2.24 |
| range (only outside p10-p90), strength 1.0 | 2.62 -> 2.54 | 0.72 |
| range, strength 0.6 | 2.62 -> 2.52 | 0.43 |

Natural track-to-track variation (3.6 dB) exceeds the damage from a typical EQ mistake (2.6 dB), so
blind EQ undo is ill-posed: median matching hurts more than it helps. The same ambiguity explains why
the joint network made EQ'd inputs worse in E1 (9.9 -> 9.1 dB). Range mode became the default: it
leaves normal tracks nearly untouched and trims only outliers (heavy-EQ subset: 4.75 -> 4.18).
A reference-track mode is the right tool when the user knows the sound they want.

### E3 result. Pretrained warm start vs from scratch (artifact task)
| run | steps | reverb | reverb+echo | reverb+echo+clip | noise (LSD) | identity |
|---|---:|---|---|---|---|---|
| `art_small` from scratch (7.7M) | 8000 | 2.9 -> 3.7 | 0.9 -> 2.3 | 0.5 -> 1.2 | 17.9 -> 8.6 | 117 -> 27.7 |
| `ft_bs` fine-tuned vocal BS-RoFormer (51M) | 1000 | 2.9 -> 6.4 | 0.9 -> 3.9 | 0.5 -> 3.6 | 17.9 -> 7.6 | 117 -> 29.1 |

A vocal dereverb model is useless on full mixes zero-shot (E2) but is a strong initialization:
1000 fine-tuning steps beat 8000 from-scratch steps by a wide margin. From-scratch run stopped;
the GPU goes to `ft_bs`.

### E6. Which stage fixes what (end-to-end, held-out, `ft_bs` at step 1000)
`python -m remaster.evaluate_pipeline`, 10 held-out clips of 12 s. SI-SDR dB / log-spectral distance dB.

| condition | input | de-echo only | network only | de-echo + network |
|---|---:|---:|---:|---:|
| reverb | 1.4 / 5.7 | 1.4 / 5.7 | 7.1 / 4.8 | 7.1 / 4.8 |
| echo | 10.5 / 5.0 | 33.1 / 1.4 | 10.2 / 5.0 | 25.6 / 1.6 |
| reverb+echo | -1.6 / 8.3 | -0.9 / 7.2 | 4.1 / 7.7 | 5.3 / 6.5 |
| clip | 18.9 / 11.5 | 18.9 / 11.5 | 18.1 / 7.3 | 18.1 / 7.3 |
| noise | 33.9 / 12.7 | 33.9 / 12.7 | 26.1 / 5.0 | 26.1 / 5.0 |
| identity | 120.2 / 0.0 | 120.2 / 0.0 | 29.1 / 0.5 | 29.1 / 0.5 |

- The stages are complementary: DSP owns discrete echo, the network owns reverb.
- The network perturbs clean audio at about -29 dB, which caps anything it touches (echo drops
  from 33.1 to 25.6 after the network). Fix: `enhance` bypasses the network when its proposed change
  is below -24 dB relative to the input. Longer training with more pass-through examples should
  lower that floor.
- Clip and noise: spectra move much closer to clean (LSD 11.5 -> 7.3, 12.7 -> 5.0) while SI-SDR drops
  slightly, again the -29 dB floor on inputs that were already above it.
- Reverb+echo, de-echo only, barely moves SI-SDR against the clean clip because the remaining reverb
  dominates that number. Measured against the reverberant clip without echo (40 clips, 25 s), the
  cepstral stage detects 35/40 and takes SI-SDR from 9.3 to 25.6 dB, so it does work under reverb.

### E3 continued. `ft_bs` validation history
From step 2000 on, 40% of training samples come from lossless MUSDB18-HQ (train split, mixtures and
stems) and 15% of samples are clean pass-through.

| step | reverb | echo | reverb+echo | reverb+echo+clip | clip | noise | identity |
|---:|---|---|---|---|---|---|---|
| 1000 | 2.9 -> 6.4 | 9.5 -> 9.6 | 0.9 -> 3.9 | 0.5 -> 3.6 | 19.6 -> 18.2 | 31.0 -> 26.3 | 29.1 |
| 2000 | 2.9 -> 7.0 | 9.5 -> 10.4 | 0.9 -> 4.5 | 0.5 -> 4.2 | 19.6 -> 18.3 | 31.0 -> 26.1 | 29.3 |
| 4000 | 2.9 -> 7.1 | 9.5 -> 11.0 | 0.9 -> 5.3 | 0.5 -> 4.8 | 19.6 -> 18.9 | 31.0 -> 28.2 | 33.9 |
| 6000 | 2.9 -> 7.4 | 9.5 -> 11.6 | 0.9 -> 5.5 | 0.5 -> 5.0 | 19.6 -> 19.1 | 31.0 -> 28.8 | 34.7 |

### E6 rerun with `ft_bs` at step 6000 (same clips; network applied without the bypass gate)
| condition | input | de-echo only | network only | de-echo + network |
|---|---:|---:|---:|---:|
| reverb | 1.4 / 5.7 | 1.4 / 5.7 | 8.0 / 4.9 | 8.0 / 4.9 |
| echo | 10.5 / 5.0 | 33.1 / 1.4 | 12.8 / 4.3 | 28.9 / 1.6 |
| reverb+echo | -1.6 / 8.3 | -0.9 / 7.2 | 4.7 / 7.4 | 5.9 / 6.7 |
| clip | 18.9 / 11.5 | 18.9 / 11.5 | 18.3 / 6.5 | 18.3 / 6.5 |
| noise | 33.9 / 12.7 | 33.9 / 12.7 | 29.4 / 5.1 | 29.4 / 5.1 |
| identity | 120.2 / 0.0 | 120.2 / 0.0 | 36.2 / 0.4 | 36.2 / 0.4 |

Reverb gain grew from +5.7 to +6.6 dB and the clean-audio floor from 29 to 36 dB between steps 1000 and 6000.

## 2026-10-02

### E7. Reverb meters for the decision layer
`remaster/reverb_meter.py`. Wetness = level of what a dereverb model removes relative to what it keeps.
61 MUSDB18-HQ train songs, 20 s vocal-active excerpts, reverb added at known direct-to-reverberant ratios.
Raw rows: `docs/evidence/reverb_calibration.json`.

| reading (dB), p10 / p50 / p90 | as produced | +reverb DRR 12 | DRR 6 | DRR 0 |
|---|---|---|---|---|
| full-mix model on mixtures (excess meter) | -44.1 / -41.2 / -32.7 | -35.1 / -20.4 / -15.4 | -13.9 / -9.4 / -7.6 | -5.7 / -3.6 / -2.7 |
| vocal model on vocal stems (absolute meter) | -32.5 / -8.3 / -1.3 | -12.1 / -7.6 / -0.6 | -6.1 / -3.8 / 1.8 | -0.5 / 1.1 / 6.9 |

- The full-mix model is a clean excess-reverb meter: readings order correctly in 97-100% of songs and
  produced mixes sit about 20 dB below even light added reverb. It reads produced vocal stems at
  -43.7 dB, so it leaves production reverb alone.
- The vocal model reads absolute wetness. Produced vocals span a wide range, which is what makes
  "too wet" and "too dry" definable, and also means produced and over-wet vocals overlap.

### E8. Full-mix model vs stems vs two-level auto (MUSDB18-HQ test, 16 songs, 20 s)
`python -m remaster.evaluate_stems`. SI-SDR dB against the clean mix; no mastering, no de-echo.

| scenario | input | full-mix model | stem pipeline (Demucs + per-stem) | auto, no guards |
|---|---:|---:|---:|---:|
| room reverb on whole mix | 4.4 | 8.8 | 5.2 | 8.7 |
| reverb on vocal only | 6.2 | 7.0 | 8.3 | 8.8 |
| clean | (124.2) | 117.3 | 106.1 | 103.5 |

- Room reverb is best handled on the whole mix; the excess meter reads much lower on separated stems,
  so per-stem removal is too timid.
- Vocal-only reverb is best handled on the vocal stem.
- Auto = whole-mix excess decision, then vocal-stem decision; stems are only remixed if the vocal changes.
- Clean-input failures without guards, 3 of 16: one false room detection (14.8 dB), one vocal stem read
  +14.3 dB wet and was gutted (6.4 dB), one vocal at the edge of normal was trimmed (19.8 dB).
  Guards added: soft gate on the room decision, vocal readings above +8 dB treated as not credible,
  vocal trigger moved from the corpus p90 (-1.3 dB) to 0 dB.

Rerun with guards, 24 songs (`docs/evidence/stems/stems_eval.md`):

| scenario | input | full-mix model | stem pipeline | auto with guards |
|---|---:|---:|---:|---:|
| room reverb on whole mix | 4.7 | 9.1 | 5.3 | 9.0 |
| reverb on vocal only | 6.7 | 7.9 | 8.0 | 8.7 |
| clean | (124.1) | 119.5 | 111.7 | 114.9 |

Auto decisions: room reduced in 24/24 room songs and 1/24 clean songs; vocal reduced in 9/24 wet-vocal
songs and 1/24 clean songs. The +14 dB vocal reading is now reported as not credible and left alone.
Two clean songs are still altered: one room false positive (14.8 dB) and one vocal judged too wet
(14.0 dB). Both are cases where the meter reads a finished production as excessive, which no threshold
separates cleanly; the reverb target slider is the user-side remedy.

### E9. VU-style gain riding
`remaster/vu.py`. Meter check: a -18 dBFS sine reads 0.0 VU and is within 0.2 dB of steady 300 ms after onset.
40 held-out clips with injected section-level errors (7 s sections, -9..+6 dB):

| setting (ratio / dead zone / window / max) | section level error in -> out (dB RMS) | max ride on clean clips (mean, p90) |
|---|---|---|
| 0.75 / 1.5 dB / 3 s / 6 dB (default) | 4.04 -> 2.38 | 1.03, 2.55 |
| 1.0 / 1.5 / 3 / 6 | 4.04 -> 2.09 | 1.28, 3.40 |
| 1.0 / 1.0 / 3 / 9 | 4.04 -> 1.76 | 1.79, 3.89 |
| 1.0 / 1.0 / 2 / 9 | 4.04 -> 1.75 | 2.14, 4.45 |

Level error improved on 40/40 clips at the default; SI-SDR against the unmodified clip 7.9 -> 12.1 dB.
More aggressive settings fix more and also ride clean, dynamic tracks harder, so the default stays moderate
and the amount is a UI control.

### Data gallery
`docs/evidence/dataset_samples/`: 24 labeled training pairs (audio, parameters in `index.json`, `gallery.png`).

### E10. What EnCodec representations know about degradations
`remaster/token_probe.py`. 32k training / 3.2k held-out 4 s clips (FMA-medium), each effect present with
probability 0.3. Identical probes (4-layer transformer, 256-d) on different inputs. Measured shapes for a 4 s
clip from the 48 kHz stereo EnCodec at 24 kbps: tokens [16, 598], encoder latents [128, 598]; log-mel [128, 345].

AUROC for detecting each effect (0.5 = chance):

| input | reverb | echo | eq | comp | clip | noise | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| tokens, 16 codebooks | 0.83 | 0.55 | 0.54 | 0.56 | 0.79 | 0.80 | 0.68 |
| tokens, 8 codebooks | 0.84 | 0.55 | 0.54 | 0.57 | 0.79 | 0.80 | 0.68 |
| tokens, 4 codebooks | 0.83 | 0.55 | 0.54 | 0.59 | 0.78 | 0.78 | 0.68 |
| tokens, 1 codebook | 0.80 | 0.55 | 0.56 | 0.62 | 0.78 | 0.80 | 0.685 |
| continuous encoder latents | 0.88 | 0.59 | 0.61 | 0.76 | 0.93 | 0.91 | 0.78 |
| log-mel spectrogram | 0.75 | 0.61 | 0.61 | 0.73 | 0.98 | 0.98 | 0.778 |

Strength regression (mean absolute error; in parentheses, the error of always predicting the mean):
latents get reverb DRR to 2.8 dB (4.5) and noise SNR to 2.4 dB (6.1); tokens get 3.5 (4.5) and 5.5 (6.1).
Echo delay, EQ size and compression ratio are not recovered by either (errors at the predict-the-mean level).

Readings:
- Quantization throws away most of what distinguishes a degraded signal. Latents beat tokens on every
  effect, most on clipping (0.93 vs 0.79), noise (0.91 vs 0.80) and compression (0.76 vs 0.56).
- Codebooks 2-16 add nothing for this purpose: one codebook does as well as sixteen. For this probe the fine
  residual codebooks carried no usable information about the degradation. They were 15/16 of what the
  token model was trained to predict.
- Echo and EQ are near chance from any EnCodec representation at this probe size. Echo is found reliably
  by the cepstral detector (E4); EQ is ambiguous by nature (E5).
- A plain log-mel ties the latents overall (0.778 vs 0.78). The latents are better at reverb (0.88 vs 0.75),
  the mel at clipping and noise (0.98 vs 0.93 / 0.91; noise SNR error 1.6 dB vs 2.4). So the EnCodec encoder
  is a reasonable analysis front end, mainly for reverb, and its discrete tokens are the weakest option tested.
- Caveat: one probe architecture, one training budget (6 epochs). A bigger probe could move the numbers.

### E11. Reference-free quality check (Audiobox Aesthetics)
`python -m remaster.evaluate_quality`, checkpoint `study/base` at step 15000. 24 held-out FMA clips of 12 s, all
level-matched before scoring. PQ = production quality, CE = content enjoyment, both 1-10, predicted with no
reference. Restoration only (no mastering, no stems). Raw rows: `docs/evidence/quality/quality.json`.

| condition | PQ clean | PQ damaged | PQ restored | clips where restored > damaged | CE damaged | CE restored |
|---|---:|---:|---:|---:|---:|---:|
| reverb | 6.66 | 6.37 | 6.63 | 79% | 5.36 | 5.81 |
| echo | 6.66 | 6.44 | 6.59 | 75% | 5.38 | 5.59 |
| reverb+echo | 6.66 | 6.37 | 6.59 | 79% | 5.30 | 5.67 |
| clip | 6.66 | 6.45 | 6.43 | 46% | 5.49 | 5.47 |
| noise | 6.66 | 6.45 | 6.44 | 12% | 5.55 | 5.53 |
| clean | 6.66 | 6.66 | 6.64 | 42% | 5.66 | 5.63 |

- Reverb and echo: an independent model that never saw the clean reference rates the restored audio
  almost as high as the clean original. This agrees with the SI-SDR gains.
- Clipping and noise: no improvement by this measure, even though the spectrum moves closer to clean
  (E6). The network's handling of these two is not yet an audible win and should not be claimed as one.
- Clean input is left essentially unchanged (6.66 -> 6.64).
- The music itself survives: chroma similarity to the clean clip is 0.99 or higher in every condition and
  rhythm-envelope correlation 0.97 or higher (`remaster/descriptors.py:content_similarity`).
- Caveat: a predictor is a proxy for listening. It was trained on general audio, and a 0.2-0.3 point
  spread is small on a 10-point scale.

### E12. Controller for EQ / compression / level riding: mixed training stalls, single effects learn
`remaster/controller.py` outputs an EQ curve (32 points) and a frame-rate gain trajectory, applied by DSP.
- Upper bound with ideal parameters computed from the clean signal (32 val tracks, 8 s): EQ tonal error
  2.67 -> 0.31 dB, compression envelope error 1.27 -> 0.04 dB, level riding 3.67 -> 0.01 dB. The
  parameterization can express nearly all of the repair.
- Trained on the full mix of effects from the start, with audio loss alone or with direct parameter
  supervision, the network stays at "do nothing" for 3000+ steps (validation identical to the input).
- Trained on level riding alone it generalizes within 2000 steps: envelope error 3.67 -> 1.51 dB,
  LSD 3.11 -> 1.12. Without clean negatives it also flattens tracks that had no problem (identity env 0 -> 0.70).
- Reading: the mixed problem has a "do nothing" optimum the network cannot leave, because it cannot yet
  tell an injected fault from natural dynamics. Running now: per-effect specialists trained against clean
  negatives, and a curriculum (single effect -> pairs -> full mix, advance on plateau, LR stepped down).

### E13. Reverb the model never saw
Added `degrade.algo_rir`: a Schroeder/Moorer comb-and-allpass reverb (the sound of reverb plug-ins), built in
the frequency domain. Before training on it, used it as a held-out family. Checkpoint `study/base` step 15000,
40 held-out FMA clips, 6 s, DRR -3..9 dB (`docs/evidence/unseen_reverb.json`).

| reverb source | SI-SDR in -> out | LSD in -> out |
|---|---|---|
| MIT real room IRs (the same 270 IRs used in training) | 3.7 -> 11.4 | 2.8 -> 2.0 |
| noise-tail synthetic (family seen, new draws) | 3.7 -> 9.1 | 6.3 -> 4.9 |
| algorithmic comb/allpass (never seen) | 3.6 -> 8.7 | 7.8 -> 5.8 |

- It generalizes to an unseen reverb family almost as well as to the synthetic family it trained on (+5.1 vs +5.4 dB).
- The real-room number is optimistic: those exact impulse responses were in training. A second real IR set
  kept for evaluation only is still needed (the EchoThief download link I tried is dead).
- From the next training segment on, a share of training reverb comes from `algo_rir` (`--p-algo`).

### E14. Curriculum vs mixed-from-start for the controller (running)
Same controller, data and Huber parameter loss; only the schedule differs. Curriculum (Bengio et al. 2009):
single effect -> one or two effects ->
full mix, advancing when validation plateaus after a minimum stay, LR x1.0 / 0.6 / 0.4.

Before this, with an L1 parameter loss, nothing learned in any setup: the EQ-only specialist (7000 steps)
and the curriculum's single-effect stage (5000 steps) both output exactly zero correction. L1 pulls every
frame that needs no correction toward zero with constant force, which pins the network at "do nothing".
Huber (quadratic within 6 dB) removed that.

Step 2000 (32 val tracks, 8 s):

| run | ride: env error | ride: LSD | eq+comp+ride: LSD | eq: tonal | comp: env | identity: LSD |
|---|---|---|---|---|---|---|
| input | 3.67 | 3.11 | 5.38 | 2.67 | 1.27 | 0.00 |
| mixed from start | 3.67 | 3.10 | 5.38 | 2.67 | 1.27 | 0.01 |
| curriculum, stage 1 | 2.33 | 1.94 | 4.87 | 2.66 | 1.31 | 0.21 |
| oracle parameters | 0.01 | 0.03 | 1.47 | 0.31 | 0.04 | 0.01 |

The curriculum run has left the do-nothing solution and the mixed run has not. EQ and compression are
not yet corrected by either; the curriculum run also perturbs clean input slightly (0.21).

Later readings of E14 (curriculum in its final stage at step 15000, mixed at 17000):

| run | mean LSD over conditions | ride: env error | identity: LSD | eq: tonal | comp: env |
|---|---|---|---|---|---|
| input | 2.83 | 3.67 | 0.00 | 2.67 | 1.27 |
| mixed from start | 2.505 | 1.76 | 0.43 | 2.59 | 1.26 |
| curriculum | 2.370 | 1.52 | 0.15 | 2.62 | 1.20 |

With the Huber loss the mixed run did start learning (around step 5000-10000), later and less cleanly than
the curriculum run, which fixes more and disturbs clean input about a third as much. Neither learned EQ or
compression. The curriculum checkpoint is the one shipped as `checkpoints/controller.pt`.

Against the DSP rider (`vu.ride_gain`) on the same kind of clips: default settings reach env error 2.53 at
identity disturbance 0.11; aggressive settings reach 1.75 at 0.36. The learned controller reaches 1.52 at 0.15.
On 20 full 30 s clips (windows stitched by `controller.ride_with_controller`): input 4.52, controller 2.27,
VU rider 2.94; controller better on 14/20; clean clips disturbed by 0.44 (controller) vs 0.31 (VU).
Stitching needed one fix: each window's gain is only defined up to a constant, so windows are offset to agree
on their overlap (without that the controller scored worse than the VU rider on full tracks).

### E15. Undoing compression and limiting: not solved
- Compression specialist (`tone:comp`, full-rate gain head, clean negatives), step 2000: envelope error
  1.27 -> 1.19 dB while clean input is disturbed by 0.26. No useful learning.
- De-limiter specialist (`tone:limit`: 6-16 dB of drive into the brickwall limiter, 70% MUSDB), step 4000:
  1.78 -> 1.66 dB, clean disturbed by 0.40. No useful learning.
- Published De-limiter (Jeon & Lee 2023, MIT, their released weights) on 20 MUSDB18-HQ test excerpts limited
  with this repo's limiter (mean drive 11.5 dB): crest factor 10.1 -> 12.6 dB (clean 16.6), envelope error
  2.02 -> 1.94 dB, SI-SDR 11.7 -> 12.1 dB (`docs/evidence/delimiter_eval.json`). It restores some peaks but
  barely moves the waveform toward the original. It was trained on one commercial limiter at about -8 LUFS;
  this limiter and these drive levels are outside that.
- Reading: finished music is already compressed, so moderate extra compression has no reliable signature,
  and heavy limiting does not transfer across limiter designs. A learned de-limiter would need the sample-rate
  gain formulation of that paper trained on several limiters. Until then the pipeline measures dynamics and
  does not claim to undo them.
- The controller's 32-point EQ output also stayed near zero in every run, consistent with E5: blind EQ
  correction is ambiguous without a reference or a content-aware prior.

Safeguards on the learned level rider (clamp to +-6 dB, hold the gain through passages more than 18 dB below
the track's usual level, light smoothing), same 20 full clips: input 4.52, controller 2.36, VU rider 2.94;
controller better on 14/20; clean clips disturbed by 0.30 (controller) vs 0.31 (VU); largest ride on a clean
clip 5.8 dB, mean 1.1 dB. Before the safeguards one clean clip was ridden by 8.1 dB during a quiet passage.

### E16. Dynamics decision in the mastering stage
`master.dynamics_decision`. Norms from 600 FMA-small clips (`remaster/dynamics_norms.json`): peak-to-loudness
ratio (true peak minus integrated LUFS) p10 8.1, median 11.4, p90 15.7 dB; crest factor p10 10.0, median 13.6,
p90 18.1 dB; integrated loudness median -11.9 LUFS.
- Inside 8.1-15.7 dB: left alone.
- Above: 2:1 soft-knee bus compression aimed at the excess, capped at 4 dB. On a test clip with added
  transients, PLR 18.2 -> 17.5 dB before the limiter (2.9 dB of gain reduction).
- Below: flagged as already heavily compressed, and the limiter is not allowed to add more than 1 dB.
The rule is measured rather than learned, because E15 showed undoing compression is not learnable here yet.

### E17. Does time attention lock onto the echo delay? (30 clips)
`python -m remaster.attention_probe`, checkpoint at step 15000. Each held-out clip gets one echo with a random
delay between 90 and 500 ms and gain 0.3 to 0.6, with no reverberation. The lag profile of time attention (mean
weight against frames back, over every fourth band and all heads) is compared with the profile on the same clip
without the echo. Raw rows: `docs/evidence/attention_lag.json`.

| layer | largest relative increase within one frame of the true delay | median attention ratio at the true lag |
|---|---:|---:|
| 1 | 9 / 30 | 1.00 |
| 6 | 30 / 30 | 2.50 |
| 12 | 26 / 30 | 14.3 |

The first layer does not respond to the echo at all. By layer 6 the network has located it in every clip, and by
layer 12 it puts fourteen times more weight on the frame one echo-delay back than it does without the echo. The
delay is never an input. One trace with attention maps is in `docs/figures/stage_4_attention.png`.

### E18. Head-to-head on reverberant full mixes: WPE, the released vocal model, Lacquer
`python -m remaster.evaluate_baselines`, 20 held-out FMA clips of 12 s, DRR -3 to 9 dB, reverb drawn from all
three families (7 measured rooms, 9 noise-tail, 4 algorithmic). WPE is `nara_wpe` with 30 taps, delay 2, 3
iterations on a 2048/512 STFT. Checkpoint at step 15000. Raw rows: `docs/evidence/baselines/baselines.json`.

| method | SI-SDR (dB) | LSD (dB) |
|---|---:|---:|
| damaged input | 3.2 | 5.48 |
| WPE (classical, no training) | 3.0 | 5.15 |
| vocal BS-RoFormer as released | 2.9 | 13.54 |
| Lacquer restoration network | **9.1** | **4.17** |

By reverb family (SI-SDR): measured rooms 2.9 -> WPE 1.9, vocal 0.5, Lacquer 11.3; noise-tail 3.1 -> 4.8, 6.1,
8.4; algorithmic 4.0 -> 0.9, 0.0, 6.9. Lacquer beats WPE on 18 of 20 clips. WPE nudges the spectrum closer to
clean and does not improve the waveform on music at these settings. The released vocal model helps on some
noise-tail clips and is destructive on the rest. Measured rooms were seen in Lacquer's training, so that column
favors it.

### E19. Musical descriptors: does restoration move the music back toward the original?
`python -m remaster.evaluate_musicality`, 16 held-out FMA tracks of 12 s, five damage conditions, restoration
stages only (no stems, no level riding, no mastering). Descriptors come from signal analysis (onset envelope,
chroma, spectral shape). "Gap closed" is `1 - |restored - clean| / |damaged - clean|` averaged per clip. Full
tables: `docs/evidence/musicality/musicality.md`.

| condition | same notes (chroma similarity to clean) | same rhythm (onset-envelope correlation) | note decay (s), clean 0.28 | brightness gap closed | dynamics gap closed |
|---|---:|---:|---:|---:|---:|
| reverb | 0.981 -> 0.993 | 0.949 -> 0.971 | 0.37 -> 0.27 | 63% | 19% |
| echo | 0.989 -> 0.995 | 0.912 -> 0.986 | 0.36 -> 0.33 | 16% | 85% |
| reverb + echo | 0.978 -> 0.991 | 0.854 -> 0.959 | 0.38 -> 0.35 | 58% | 64% |
| clipping | 0.998 -> 0.997 | 0.950 -> 0.948 | 0.50 -> 0.49 | -9% | -18% |
| noise | 0.999 -> 0.998 | 0.969 -> 0.969 | n/a | -30% | 82% |

Reverb smears onsets and lengthens note decay; the restored clips get both back (decay 0.37 s to 0.27 s against
0.28 s clean, on the 14 clips where a decay can be measured). Echo damages the rhythm descriptor most, and the
echo stage recovers it (0.912 to 0.986). Clipping and noise are not improved by the network on any descriptor,
which matches the flat SI-SDR and the quality predictor in E11. That gap is what E21 addresses for clipping.

### E20. Community dereverberation models on the same 20 clips
`python -m remaster.evaluate_external` writes the E18 clips to disk, each tool processes them, and the outputs are
scored with the same code. Models were run with `audio-separator` 0.47 at default settings, taking the dry stem.

| method | SI-SDR (dB) | LSD (dB) |
|---|---:|---:|
| damaged input | 3.2 | 5.48 |
| WPE | 3.0 | 5.15 |
| vocal BS-RoFormer (anvuew), as released | 2.9 | 13.54 |
| Reverb HQ (MDX-Net, FoxJoy) | 3.9 | 12.75 |
| UVR-DeEcho-DeReverb (VR arch, FoxJoy) | 3.4 | 11.15 |
| MDX23C De-Reverb (aufr33, jarredou) | -0.5 | 12.92 |
| Lacquer restoration network | **9.1** | **4.17** |

The MDX-Net model is the best of the released tools on full mixes (+0.7 dB) and all of them raise the
log-spectral distance, mostly by thinning the mix. The Mel-Band "de-reverb-echo v2" model (Sucial) returns a
near-silent dry stem on full mixes (-55.9 dB) and is left out of the table. These tools were built for vocal
stems, so the comparison shows what happens when they are pointed at a finished mix. The clips include measured
rooms from Lacquer's training set; E22 repeats the comparison on rooms no model here has seen.

### E21. Clipping: a DSP declipper where the network had no effect
The network never improved clipped clips in SI-SDR (E5, E11, E19). Hard clipping leaves exact knowledge behind:
samples under the ceiling are untouched and samples at the ceiling were at least that large. `remaster/declip.py`
detects a flat ceiling on both polarities of a channel and rebuilds the clipped samples with A-SPADE (Kitic,
Bertin, Gribonval 2015): per frame, the sparsest spectrum whose waveform agrees with both facts. 2048-sample
frames, hop 512, DFT redundancy 2, relative tolerance 0.1.

`python -m remaster.evaluate_declip`, 29 held-out FMA tracks of 8 s, clipping threshold at the 90th to 99.7th
percentile of sample magnitude (the training range). Raw rows: `docs/evidence/declip/declip.json`.

| hard clipping | SI-SDR (dB) | LSD (dB) |
|---|---:|---:|
| clipped input | 20.1 | 11.4 |
| restoration network | 19.9 | 4.2 |
| declipper | **26.7** | 11.0 |
| declipper, then network | 24.7 | **3.9** |

The declipper improves every clip (29 of 29, median +6.2 dB). The two methods fix different things: the declipper
restores the waveform peaks, the network removes the distortion products above the music's bandwidth (which is
what the log-spectral distance sees) and gives back about 2 dB of waveform accuracy doing so. Detection: 29 of 29
hard-clipped clips, 0 of 29 clean clips, and 0 of 150 full-length MUSDB18-HQ mixtures. Tanh saturation at the
same thresholds has no flat ceiling, is not detected (0 of 29) and passes through unchanged (17.4 dB in and out),
so soft saturation remains unsolved. A clipped file that was later resampled or lossy-encoded also loses its flat
ceiling and will not be detected. Cost: about 1.2x real time on CPU for heavily clipped audio.

### E22. Rooms nobody trained on: the Aachen impulse response database
E18's measured rooms were in the training set. This test uses rooms the network has never seen: 67 binaural
impulse responses from six real rooms (booth, office, meeting room, lecture room, stairway, Aula Carolina) of the
Aachen Impulse Response database v1.4 (Jeub, Schafer, Vary 2009), dummy-head recordings converted to stereo.
`python -m remaster.evaluate_baselines --rir data/raw/air/wav --p-real 1.0 --tracks 40`: 40 held-out FMA clips of
12 s, DRR -3 to 9 dB, checkpoint at step 15000. Raw rows: `docs/evidence/heldout_rooms/`.

| method | SI-SDR (dB) | LSD (dB) |
|---|---:|---:|
| damaged input | 3.1 | 4.44 |
| WPE | 0.9 | 4.58 |
| vocal BS-RoFormer (anvuew), as released | 0.1 | 13.80 |
| Reverb HQ (MDX-Net, FoxJoy) | 3.2 | 12.45 |
| UVR-DeEcho-DeReverb (VR arch, FoxJoy) | 3.1 | 9.94 |
| MDX23C De-Reverb (aufr33, jarredou) | 0.5 | 12.40 |
| Lacquer restoration network (step 15000) | **4.8** | **4.08** |

Lacquer improves all 40 clips and is the only method that improves the average, but the gain is +1.7 dB (median
+1.3), far below the +8.4 dB on rooms from its training set (E18). By reverb level: +2.7 dB at DRR -3 to 1,
+1.5 dB at 1 to 5, +0.8 dB at 5 to 9. The 270 training rooms are short (median T30 0.35 s, 90th percentile 1.0 s)
and mono; the Aachen rooms are longer (median 0.93 s) and binaural. The honest reading is that the network
learned the training rooms' early reflection patterns well and general room reverb much less well. The number to
quote for real rooms is this one. E24 retrains with 4000 simulated rooms to close the gap.

### E23. Does CBAM or FiLM help? A spectrogram U-Net ablation
`scripts/ablate_unet_lab.sh`: a complex-spectrogram U-Net (`remaster/unet.py`) trained from scratch for 8000
steps on the same task, data and losses as the from-scratch band-split run of E2 (FMA-medium, artifact task,
batch 4, lr 3e-4), in three variants. Validation on the same fixed set.

| model (8000 steps, from scratch) | mean SI-SDR gain | reverb | echo | noise | identity |
|---|---:|---:|---:|---:|---:|
| U-Net | -1.42 | 3.0 -> 4.1 | 9.5 -> 8.9 | 31.0 -> 22.8 | 25.4 |
| U-Net + CBAM | -0.86 | 3.0 -> 4.0 | 9.5 -> 9.1 | 31.0 -> 25.1 | 30.0 |
| U-Net + CBAM + FiLM | -0.92 | 3.0 -> 4.0 | 9.5 -> 9.1 | 31.0 -> 24.9 | 28.9 |
| band-split transformer, 7.7 M (E2) | | 2.9 -> 3.7 | 9.5 -> 8.9 | | 27.7 |
| fine-tuned BS-RoFormer, 1000 steps (E3) | | 2.9 -> 6.4 | 9.5 -> 9.6 | | 29.1 |

CBAM helps the U-Net leave clean audio alone (identity 25.4 to 30.0 dB) and costs nothing on reverb. FiLM with
no conditioning signal is a learned per-channel scale and shift, and adds nothing measurable (-0.06 dB). Neither
changes the main picture: every from-scratch model at this budget gains about 1 dB on reverb, has a negative
mean gain because it damages the lightly degraded conditions, and is far behind 1000 steps of fine-tuning from
the pretrained checkpoint. Attention modules matter much less here than the initialization.

### E24. What released music measures like, by genre, against unmastered mixes
`python -m remaster.build_mastering_norms` measures 60 mastering features per track (`remaster/analysis.py`:
BS.1770 loudness, loudness range, true peak, peak-to-loudness ratio, third-octave spectrum, spectral slope,
side/mid level in five bands, channel correlation overall and below 150 Hz, crest factor and level spread in
four bands) on 24,980 FMA-medium tracks with their genre labels and on the 150 MUSDB18-HQ mixtures and their
stems. Norms use training-split tracks only (`remaster/mastering_norms.json`). Figure:
`docs/figures/mastering_corpus.png`.

| group | n | loudness (LUFS) | peak-to-loudness (dB) | spectral slope (dB/oct) | side minus mid (dB) |
|---|---:|---:|---:|---:|---:|
| all released | 24,474 | -11.9 [-18.9, -7.5] | 11.6 [7.9, 15.8] | -5.1 [-7.5, -3.5] | -11.3 [-28.0, -4.1] |
| Hip-Hop | 2,141 | -10.5 [-15.9, -7.3] | 10.7 [8.0, 14.1] | -4.6 [-5.8, -3.6] | -15.1 [-26.1, -8.3] |
| Rock | 6,964 | -10.7 [-16.6, -6.7] | 10.5 [7.2, 14.6] | -5.0 [-6.5, -3.8] | -10.7 [-23.7, -5.0] |
| Electronic | 6,173 | -11.3 [-17.0, -7.6] | 11.4 [8.3, 15.2] | -4.9 [-7.0, -3.5] | -11.4 [-25.0, -4.5] |
| Jazz | 377 | -14.2 [-21.6, -9.9] | 13.3 [10.1, 17.7] | -6.1 [-8.6, -3.9] | -9.3 [-21.9, -3.3] |
| Classical | 612 | -20.6 [-28.5, -14.5] | 14.9 [11.9, 18.0] | -7.3 [-11.6, -4.7] | -4.1 [-15.6, -0.5] |
| unmastered professional mixes (MUSDB18-HQ) | 150 | -15.8 [-17.8, -13.6] | 15.8 [13.7, 17.9] | -5.1 [-6.2, -4.4] | -10.0 [-16.1, -6.2] |

Median [10th, 90th percentile]. Two readings matter for the design. First, unmastered professional mixes have
the same tone and stereo image as released music (slope -5.1 against -5.1 dB/octave) and differ in dynamics:
4 dB more peak-to-loudness ratio. Mastering a good mix is mostly a dynamics and loudness job. Second, the spread
inside released music is wide (tone varies by 6 dB per band between tracks), while professional mixes are tight
(2.9 dB). Instrument balance in the professional mixes, as stem loudness relative to the mix: vocals -3.5 LU
[-6.1, -1.8], other -4.9, drums -6.4, bass -7.6.

### E25. Can a tonal or stereo fault be corrected without a reference? Mostly no, and here is the bound
Two tests. (a) `python -m remaster.evaluate_mastering`: 200 held-out released tracks get a mastering fault
(spectral tilt of 0.6 to 1.6 dB/octave, one or two broad EQ bumps of 3 to 8 dB, side level changed by 3 to 9 dB,
upward expansion) and go through the mastering chain (`remaster/mastering.py`) with different targets. Error is
measured against the original track's own features. (b) `python -m remaster.tone_identifiability` works on the
measured spectra directly.

| fault | measure (dB) | damaged | global norms | genre norms | reference mode (the original as reference) | Matchering 2.0 (same reference) |
|---|---|---:|---:|---:|---:|---:|
| tilt | tone error | 3.13 | 3.03 | 2.98 | **0.19** | 0.72 |
| EQ bumps | tone error | 2.56 | 2.53 | 2.48 | **0.12** | 0.65 |
| stereo width | width error | 5.68 | 5.09 | 4.93 | 0.27 | **0.03** |
| over-dynamic | band crest error | 0.61 | 0.52 | 0.54 | **0.32** | 0.52 |
| any of the above | peak-to-loudness error | 1.14 | 1.29 | 1.25 | 0.44 | **0.21** |
| no fault | tone moved | 0 | 0.65 | 0.66 | 0.05 | n/a |

Numbers are for the final chain (blind moves capped at 1.5 dB; reference mode matches the mid and side spectra
separately). Population norms take back 3% of a tilt and 1% of an EQ bump, genre labels add 2 points, and an
undamaged track is moved by 0.65 dB. With the original as reference the same chain removes 94 to 95% of a tonal
fault and 95% of a width fault. The reason is in (b): a held-out track's own tone is 5.2 dB RMS away from the
population mean, 4.6 dB from its genre mean and 4.3 dB from the mean of its 20 nearest neighbours in
EQ-invariant descriptors, while the fault is 2.3 dB. A Gaussian posterior-mean estimator with the full band
covariance does no better than the range rule (2.28 to 2.25 dB). Professional mixes are tighter (2.9 dB band
spread), and there the range rule removes 14% (2.41 to 2.08 dB) at 0.4 dB disturbance. Stems do not rescue
this: per-stem tone varies more than the mix (slope spread 1.6 dB/octave for vocals against 0.8 for the mix),
although stem deviations are nearly independent of each other (correlations 0.07 to 0.25).

Against Matchering 2.0 given the same reference (200 tracks per fault): our reference mode ends closer in tone
after a tilt (0.19 against 0.72 dB, better on 80% of tracks) and after EQ bumps (0.12 against 0.65 dB, better on
88%), and closer in band dynamics. Matchering is closer on stereo width (0.03 against 0.27 dB; our median is
0.06) and on peak-to-loudness ratio (0.21 against 0.44 dB), the second because our output is held at 0 dBTP when
the reference peaks above full scale. An earlier version of the reference mode with a five-band width control
left 0.41 dB of width error and 0.44 dB of tilt error; matching mid and side spectra separately fixed both.
Consequence for the system: blind tone and width corrections stay small, and anything stronger needs a
reference track.

### E26. A reference-free quality predictor does not see mastering faults
`python -m remaster.probe_quality_sensitivity`: 100 held-out tracks, each scored by Audiobox Aesthetics as is and
with the faults of E25 (two draws each, level-matched).

| fault | production-quality change | original scored higher |
|---|---:|---:|
| tilt | -0.02 | 52% |
| EQ bumps | -0.06 | 60% |
| stereo width | +0.00 | 47% |
| over-dynamic | +0.00 | 53% |

The predictor that separates reverberant from clean clips (E11) is close to chance on mastering faults. It
cannot steer a mastering search and it is not evidence for or against a mastering result, so mastering is
evaluated here with feature errors against known originals and with baselines at equal loudness.

### E27. Retraining with simulated rooms to close the unseen-room gap (running)
E22 showed the network had learned its 270 training rooms better than reverberation in general. Fix under test:
`python -m remaster.build_ism_bank` generates 4000 stereo impulse responses of shoebox rooms with
pyroomacoustics (image sources for the early part, ray tracing for the tail, frequency-dependent absorption,
a spaced microphone pair, RT60 0.25 to 2.8 s). The fine-tune continues from step 16500 with these added as a
second impulse-response source (`--train-rir`, sampled as often as the measured rooms; run `ft_bs2`). Validation
keeps its fixed set, and the Aachen rooms stay unseen. Same 40 Aachen clips as E22, and 20 clips with 64
simulated rooms from seeds not used in training.

| reverb source (SI-SDR, dB) | input | step 15000 (before) | step 19000 (2500 steps with simulated rooms) | paired change |
|---|---:|---:|---:|---:|
| Aachen measured rooms, unseen (40 clips) | 3.11 | 4.81 | 5.51 | +0.70 [+0.53, +0.88], 37 of 40 better |
| simulated rooms, unseen seeds (20 clips) | 2.25 | 5.20 | 5.98 | +0.78 [+0.54, +1.04], 19 of 20 better |
| fixed validation set, training families | 3.0 | 8.2 | 8.5 | |

The gain on unseen measured rooms went from +1.70 to +2.40 dB with no loss on the validation set. Part of that
is 4000 more training steps in general; the earlier trend on the validation set was about +0.1 dB per 1000
steps, so most of it is the new data. The run continues to 60000 steps and this entry will be updated.

### E28. Loudness without damage: limiters at equal loudness and equal true peak
`python -m remaster.evaluate_loudness --true-peak-safe`: the loudest 30 s of each of the 50 MUSDB18-HQ test
mixtures (unmastered, -14.7 LUFS and 14.3 dB peak-to-loudness on these excerpts) is driven into each limiter until
the output measures the target loudness, and the limiter's ceiling is lowered until the 4x-oversampled peak
respects -1 dBTP. What the limiter did is split into a smooth gain (fitted in 10 ms windows) and the remainder:
"distortion" is the remainder relative to the signal, "gain movement" is the 5th to 95th percentile spread of the
fitted gain. A clipper scores badly on the first, a pumping limiter on the second. Raw rows:
`docs/evidence/loudness/`.

| system, target -9 LUFS at -1 dBTP | distortion (dB) | gain movement (dB) | reaches target |
|---|---:|---:|---|
| hard clip | -23.9 | 0.8 | yes |
| Matchering 2.0 limiter | -27.7 | 4.0 | yes |
| ffmpeg alimiter (5 ms attack, 50 ms release) | -37.1 | 6.0 | yes |
| ffmpeg loudnorm (one pass) | -37.0 | 6.6 | no (-10.9 LUFS) |
| Pedalboard limiter | -24.4 | 1.1 | no (does not hold the true-peak ceiling) |
| ours, first version (single lookahead stage) | -44.3 | 6.3 | yes |
| ours, two-stage, 1 dB clip stage | -37.7 | 3.6 | yes |
| ours, two-stage, 2 dB clip stage | -32.7 | 2.3 | yes |

The two-stage limiter (`remaster/master.py:limiter_v2`: a slow stage for sustained reduction, a fast lookahead
stage, and a 4x-oversampled soft clipper that takes the last 1 to 2 dB off the shortest peaks) is better than
Matchering on both measures (5 to 10 dB less distortion with less gain movement) and matches ffmpeg's limiter
on distortion with 2.5 dB less gain movement. Without the true-peak requirement every baseline overshoots the
ceiling by 0.9 to 1.9 dB at this loudness (`loudness.json`), and ours by 0.0. At the streaming target of -14 LUFS
these mixes need under 1 dB of limiting and every system except loudnorm is transparent (distortion below
-57 dB). The two measures are signal measures: which trade-off sounds best is a question for the listening test.

### E29. Against a commercial limiter: iZotope Ozone 9 Maximizer (musdb-XL)
musdb-XL (Jeon and Lee 2023, Zenodo 7041331) is the MUSDB18-HQ test set processed by the Ozone 9 Maximizer,
released as sample-wise gain ratios. `python -m remaster.evaluate_loudness --xl <ratios>` rebuilds the Ozone
output for the same 30 s excerpts as E28 (mean -7.5 LUFS, true peak +1.05 dBTP as shipped), then drives every
other limiter to Ozone's loudness and Ozone's true peak on each song. Raw rows:
`docs/evidence/loudness/loudness_ozone.json`.

| system at Ozone's loudness and true peak | distortion (dB) | gain movement (dB) | log-spectral distance to input (dB) |
|---|---:|---:|---:|
| Ozone 9 Maximizer | -25.7 | 1.4 | **0.95** |
| Matchering 2.0 limiter | -28.8 | 2.5 | 1.64 |
| ffmpeg alimiter | -38.6 | 4.3 | 1.66 |
| ours, 1 dB clip stage | -38.4 | 2.2 | 1.32 |
| ours, 2 dB clip stage | -33.6 | 1.3 | 1.20 |
| ours, 3 dB clip stage | -30.5 | **0.7** | 1.15 |

At equal loudness and peak, the two-stage limiter leaves 5 to 8 dB less unexplained distortion than Ozone at
equal or lower gain movement. Ozone changes the long-term spectrum least (0.95 dB against 1.15 to 1.32 dB),
so on that measure it is still ahead. These are three signal measures of one preset of a commercial product
whose modes are tuned by ear; they say the open limiter is in the same class, and a listening test has to say
which one sounds better. The Ozone outputs in the dataset exceed 0 dBTP by 1 dB on average, which a streaming
delivery spec would reject.

### E30. Can a fault inside one instrument be fixed from the finished mix? Level yes, tone no (first version)
`python -m remaster.evaluate_stem_master`: each of the 50 MUSDB18-HQ test songs is rebuilt from its true stems
with one fault on one stem (a level error of 4 to 9 dB, or one or two EQ bumps of 3 to 8 dB), plus an unmodified
copy. Three systems see only the mix: the mix-level tone correction, the stem engine on Demucs-separated stems,
and the stem engine on the true faulty stems (the bound set by perfect separation). In this first version the
stem engine corrected both level and tone, against norms from the true stems of the training songs.
Raw rows: `docs/evidence/stem_master/stem_master_v1.json`.

| fault | measure | input | mix-level chain | stem engine | stem engine, true stems |
|---|---|---:|---:|---:|---:|
| vocal level | balance error (dB) | 6.3 | | 4.3 | 4.1 |
| drum level | balance error (dB) | 6.4 | | 5.1 | 4.7 |
| bass level | balance error (dB) | 6.5 | | 5.2 | 4.7 |
| vocal level | SI-SDR to the intended mix (dB) | 10.8 | 10.9 | 12.2 | 12.8 |
| vocal tone | SI-SDR (dB) | 17.2 | 17.2 | 15.3 | 15.1 |
| drum tone | SI-SDR (dB) | 22.6 | 21.6 | 19.0 | 19.1 |
| bass tone | SI-SDR (dB) | 28.6 | 25.2 | 18.7 | 18.5 |
| no fault | SI-SDR (dB) | | 55.6 | 23.2 | 23.4 |

Three findings. Level errors on a stem are partly corrected from the mix alone (vocal balance error 6.3 to
4.3 dB), and separation costs little against true stems (4.3 against 4.1), because the correction is applied as
a stem difference. Tone correction per stem makes every case worse, with true stems as much as with separated
ones: instrument tone varies more between songs (5.5 dB per band for vocals) than the fault, the same bound as
E25. And the level rule fired on 24 of 50 unmodified songs: separated stems read about 1 LU lower than true
stems, and the norms came from whole songs while the test reads 30 s. Changes made after this run: stem tone
correction is off by default, and the balance norms are measured on separated stems over the loudest 30 s, the
same way a new track is measured (`remaster/build_stem_level_norms.py`). The rerun is E31.

### E31. Instrument balance with norms measured the way a new track is measured
Changes from E30: no stem tone correction, balance read over the loudest 30 s of the mix, and the normal range
taken from Demucs-separated stems of the 99 MUSDB18-HQ training mixes (`remaster/build_stem_level_norms.py`;
the same reading on 1500 released FMA tracks is stored alongside). Separated-stem ranges are wide: vocals -10.5 to
-1.1 LU (5th to 95th percentile, median -4.2), drums -14.9 to -4.2, bass -15.3 to -3.6, other -12.4 to -2.1.
Same 50 test songs, level faults of 4 to 9 dB on one stem. Raw rows:
`docs/evidence/stem_master/stem_master_v3.json`.

| all four stems may be moved | value |
|---|---:|
| faults acted on | 64 of 150 |
| balance error, all faults | 7.1 to 6.1 dB |
| balance error when the stage acted | 6.9 to 4.6 dB |
| SI-SDR to the intended mix when it acted | 10.6 to 13.2 dB |
| same with true stems instead of separated ones (vocal faults) | 6.69 against 6.77 dB balance error |
| unmodified songs altered | 16 of 50 |

| which stems may be moved (decision replayed on the recorded readings) | unmodified songs altered | faults caught | a different stem moved |
|---|---:|---:|---:|
| all four | 32% | 43% | 33% |
| all four, 2 LU margin | 10% | 14% | 11% |
| vocal only | 8% | 44% | 0% |
| vocal only, 1 LU margin | 6% | 34% | 0% |

The mechanics work: when the stage acts, the balance error falls by a third and separation costs almost nothing
against true stems. The decision is the limit. Instrument balance is a wide distribution in professional mixes,
so fewer than half of 4 to 9 dB faults leave the normal range, and with four stems checked a third of unmodified
songs has some stem outside it. The default is therefore the vocal alone, the one element with published level
norms: 8% of unmodified songs are touched, 44% of vocal faults are caught, and no other stem is moved by
mistake. Drums, bass and accompaniment are measured and reported.
