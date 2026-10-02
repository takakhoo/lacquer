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
  thesis model was trained to predict.
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
  negatives, and the thesis curriculum (single effect -> pairs -> full mix, advance on plateau, LR stepped down).

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
Same controller, data and Huber parameter loss; only the schedule differs. Curriculum is the thesis
schedule ported from `Curriculum_Tokenize_Master/token_train.py`: single effect -> one or two effects ->
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
