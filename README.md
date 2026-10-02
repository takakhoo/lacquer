# Lacquer

**Automatic restoration and mastering for finished music mixes.** Give it a stereo track and it removes echo
and excess reverb, evens out level problems, and sets loudness and peaks for release. It tells you every
decision it made and leaves alone whatever is already fine.

![A damaged mix being restored: the sweep reveals the cleaned spectrogram while the decisions appear](docs/demo/restore.gif)

*One real run on a held-out clip that was damaged with room reverb and a 163 ms echo. Left of the white line is
the restored audio, right of it the damaged input. The echo is found and inverted by DSP, the reverb is
removed by the network, the vocal reverb and the dynamics are measured and judged normal, and loudness is set for release.*

A lacquer is the disc a record's master is cut into. This project is the successor to my honors thesis
([neural-audio-restoration](https://github.com/takakhoo/neural-audio-restoration)), which tried to restore
music by predicting EnCodec tokens with a U-Net. [`docs/REVIVAL.md`](docs/REVIVAL.md) shows, with
measurements, why that approach could not improve audio and what replaced it.

## Headline

Held-out music with synthetic damage. SI-SDR is closeness to the clean track in dB (higher is better).

| test | damaged | after Lacquer |
|---|---:|---:|
| Echo, 79 clips (DSP stage alone, no training) | 8.8 dB | **25.2 dB** |
| Room reverb on the whole mix, 24 MUSDB18-HQ test songs | 4.7 dB | **9.0 dB** |
| Reverb on the vocal only, same songs | 6.7 dB | **8.7 dB** |
| Reverb from a plug-in style the model never trained on, 40 clips | 3.6 dB | **8.7 dB** |
| Level problems inside a track, 20 clips (envelope error, lower is better) | 4.5 dB | **2.4 dB** |

A separate judge that never sees the clean reference (Meta's Audiobox Aesthetics, production quality 1-10)
rates reverberant clips 6.37, the restored versions 6.63, and the clean originals 6.66.

What does not work yet, stated plainly: clipping and noise get spectrally closer to clean with no gain by that
judge; blind EQ correction and undoing compression or limiting are unsolved (measured, see below); two of 24
clean songs were still altered in the stem test; everything above is synthetic damage on real music.

Every number has its experiment, data and caveats in [`docs/experiments.md`](docs/experiments.md).

## Contents

- [How it works](#how-it-works)
- [The decisions](#the-decisions)
- [What the experiments found](#what-the-experiments-found)
- [Try it](#try-it)
- [Train and evaluate](#train-and-evaluate)
- [Layout](#layout)
- [Credits and licenses](#credits-and-licenses)

## How it works

Each problem goes to the tool that suits it. Problems with an exact answer use DSP. Problems that need a
learned prior use a network. Aesthetic choices are measured against a normal range and left alone inside it.

| stage | method | module |
|---|---|---|
| Discrete echo | Cepstral detection, then an exact recursive inverse filter | `remaster/deecho.py` |
| Room reverb, noise, clipping | Band-split transformer that predicts a complex mask on the stereo spectrogram, fine-tuned from a pretrained vocal dereverb model | `remaster/pretrained.py`, `remaster/train.py` |
| Reverb decisions | The same network is a calibrated meter for excess reverb on the mix; a vocal model measures how wet the vocal stem is | `remaster/reverb_meter.py`, `remaster/stems.py` |
| Level riding | A small controller network outputs a gain trajectory (a VU-ballistics rider is the fallback) | `remaster/controller.py`, `remaster/vu.py` |
| Tonal balance, loudness, peaks | Range-based tonal correction, BS.1770 loudness, 4x-oversampled true-peak limiter, optional reference track | `remaster/master.py` |

The restoration network never generates audio. It multiplies the input spectrogram by a mask that starts as
all ones, so anything it does not touch passes through unchanged. There is no codec or vocoder in the path.

## The decisions

`remaster/infer.py:enhance_auto` runs the chain and returns a report like the one in the animation:

1. **Echo.** A delayed copy leaves a ripple on the log spectrum, which is a sharp peak in the cepstrum. If one
   is found, the delay and echo path are read off and inverted. If none is found, nothing happens.
2. **Room reverb.** The network's proposed change is measured first, on a few windows. On clean productions it
   reads about -41 dB; with light added reverb about -20 dB; with heavy reverb about -4 dB. Above the gate the
   reverb is removed, scaled by how far above; below it the network is bypassed.
3. **Vocal reverb.** The vocal stem (Demucs) gets an absolute wetness reading. Produced vocals span a wide
   range, so the vocal is only changed when it is clearly outside it: reduced if far too wet, and a plate added
   if bone dry. Stems are remixed only when the vocal was changed.
4. **Level.** A gain trajectory is predicted for the whole track and applied as a fader move.
5. **Dynamics.** The peak-to-loudness ratio is compared with the corpus range (8 to 16 dB). Unusually peaky
   tracks get gentle 2:1 compression; tracks that are already squashed are flagged and not limited further.
6. **Finish.** Bands outside the normal tonal range are trimmed, loudness is set to the target, and peaks are
   limited at -1 dBTP. A reference track can replace the tonal and loudness targets.

A "reverb target" control shifts steps 2 and 3 drier or wetter.

## What the experiments found

Short versions. Each links to numbers in [`docs/experiments.md`](docs/experiments.md).

- **The thesis pipeline could not win.** Its "reverb" was a lowpass with +72.8 dB of gain that clipped every
  example, its "EQ" was a band-pass, gain was invisible in EnCodec tokens, and a perfect prediction decoded to
  9.6 dB SI-SDR, below the damaged input it was meant to fix ([`docs/REVIVAL.md`](docs/REVIVAL.md)).
- **A warm start mattered more than architecture.** From scratch, 8,000 steps bought +0.8 dB on reverb. Fine-tuning
  a public vocal dereverb model on full mixes bought +3.5 dB in 1,000 steps, although that model wrecks full mixes
  when used as released (E2, E3).
- **Echo is a DSP problem.** Cepstral detection plus an exact inverse beats the network by 15 dB with no training (E4).
- **Room reverb belongs to the whole mix, vocal reverb to the stem.** Neither approach wins both cases, so the
  pipeline decides at two levels (E8).
- **EnCodec tokens are a poor place to look for damage.** The same probe detects degradations with AUROC 0.68
  from tokens, 0.78 from the encoder's continuous latents, and 0.78 from a mel spectrogram. One codebook does as
  well as sixteen (E10).
- **The thesis curriculum earns its place.** Training the controller on all effects at once left it stuck at
  "do nothing". The single-effect, then pairs, then full-mix schedule from the thesis got it moving and ended
  ahead: level-envelope error 1.52 dB against 1.76 dB, with a third of the disturbance to clean input (E14).
- **Some things are not learnable blind at this scale.** Track-to-track tonal variation (3.6 dB) is larger than
  a typical EQ mistake (2.6 dB), so pulling toward an average spectrum hurts more than it helps (E5). Undoing
  compression and limiting did not train, and a published de-limiter only partly transfers to a different
  limiter (E15).

## Try it

Replay the recorded session in a terminal. No GPU, no checkpoints, only NumPy:

```bash
python -m remaster.demo replay
```

Run the web app (A/B player with loudness matching, spectrograms, VU traces, the decision list, and a stress
test that damages your upload first so you can hear the repair):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m remaster.app          # http://127.0.0.1:7860
```

The app looks in `remaster/checkpoints/` for `best.pt` (restoration), `vocal_dereverb.pt` (vocal meter) and
`controller.pt` (level riding). Checkpoints are not in the repository. Without them it runs the DSP stages only.

## Train and evaluate

```bash
# restoration network: fine-tune a released BS-RoFormer on full-mix degradations
python -m remaster.train --data <music dirs> --rir <impulse responses> --out runs/x --task artifact \
    --arch msst_bs --msst-config remaster/pretrained_cfg/anvuew_bs.yaml --init <released checkpoint>

# level controller with the curriculum schedule
python -m remaster.controller --curriculum --data <music dirs> --rir <IRs> --out runs/ctrl

python -m remaster.evaluate_pipeline --ckpt runs/x/best.pt --data <music> --rir <IRs> --out eval/pipeline
python -m remaster.evaluate_stems --musdb <musdb18hq/test> --rir <IRs> --mix-ckpt ... --vocal-ckpt ... --out eval/stems
python -m remaster.evaluate_quality --ckpt runs/x/best.pt --data <music> --rir <IRs> --out eval/quality
```

`scripts/` holds helpers for training on a remote GPU box over a shared SSH socket.

## Layout

```
remaster/                    package: models, DSP stages, training, evaluation, web app, demo
remaster/third_party/msst/   BS-RoFormer and Mel-Band RoFormer model code, vendored (MIT)
docs/REVIVAL.md              why the thesis pipeline failed, and the new design
docs/experiments.md          every experiment with numbers, including the negative results
docs/evidence/               figures and metric files behind those numbers
docs/demo/                   the recorded session and animation shown above
scripts/                     remote training helpers
```

The Python package keeps its working name, `remaster`.

## Credits and licenses

- Model code in `remaster/third_party/msst` is from
  [Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training) (MIT).
- The restoration network is fine-tuned from anvuew's `dereverb_bs_roformer` checkpoint (GPL-3.0). Weights
  derived from it carry that license.
- Stem separation uses [Demucs](https://github.com/facebookresearch/demucs) (MIT). Quality scoring uses
  Meta's Audiobox Aesthetics.
- Training data: FMA (per-track Creative Commons licenses), MUSDB18-HQ (educational use), MIT IR Survey.
  Several of these exclude commercial use.
